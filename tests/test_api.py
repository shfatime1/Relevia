"""Essential integration checks against the local Phase 1–4 data and indexes."""
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import MAX_UPLOAD_BYTES, app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.search = app.state.search
        cls.resources = (id(cls.search.text_search), id(cls.search.image_search),
                         id(cls.search.text_search.model), id(cls.search.image_search.model))

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_catalog_filters_and_pagination(self):
        first = self.client.get('/api/products', params={'page': 1, 'page_size': 10}).json()
        second = self.client.get('/api/products', params={'page': 2, 'page_size': 10}).json()
        self.assertEqual(first['total'], 1000)
        self.assertEqual(len(first['items']), 10)
        self.assertTrue(set(p['product_id'] for p in first['items']).isdisjoint(p['product_id'] for p in second['items']))
        filtered = self.client.get('/api/products', params={'category': 'digital cameras', 'min_price': 50, 'max_price': 200}).json()
        self.assertGreater(filtered['total'], 0)
        self.assertTrue(all(p['original_category'] == 'Digital Cameras' and 50 <= p['price'] <= 200 for p in filtered['items']))
        self.assertIn('Cameras & Photography', self.client.get('/api/categories').json()['main_categories'])
        self.assertEqual(self.client.get('/api/products', params={'category': 'unknown'}).json()['items'], [])
        self.assertEqual(self.client.get('/api/products', params={'min_price': 200, 'max_price': 50}).status_code, 422)
        self.assertEqual(self.client.get('/api/products', params={'page': 0}).status_code, 422)

    def test_details_images_and_missing_product(self):
        response = self.client.get('/api/products/B0001G6U4S')
        self.assertEqual(response.status_code, 200)
        product = response.json()
        self.assertEqual(product['product_id'], 'B0001G6U4S')
        self.assertTrue(product['specifications'])
        self.assertIsInstance(product['description'], list)
        image = self.client.get(product['image_url'])
        self.assertEqual(image.status_code, 200)
        self.assertTrue(image.headers['content-type'].startswith('image/webp'))
        for suffix in ('', '/recommendations', '/sentiment'):
            self.assertEqual(self.client.get('/api/products/NOT_A_PRODUCT' + suffix).status_code, 404)

    def test_text_search_and_resource_reuse(self):
        result = self.client.get('/api/search/text', params={'q': 'Canon PowerShot', 'category': 'Digital Cameras', 'limit': 5})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()['items']), 5)
        self.assertTrue(all(p['original_category'] == 'Digital Cameras' and p['ranking_reasons'] and p['image_url'] for p in result.json()['items']))
        self.assertEqual(self.client.get('/api/search/text', params={'q': 'simsiz qulaqlıq', 'category': 'unknown'}).json()['items'], [])
        self.assertEqual(self.client.get('/api/search/text', params={'q': '  '}).status_code, 422)
        self.assertEqual(self.client.get('/api/search/text', params={'q': 'camera', 'max_price': 'nan'}).status_code, 422)
        self.assertEqual(self.resources, (id(self.search.text_search), id(self.search.image_search),
                                          id(self.search.text_search.model), id(self.search.image_search.model)))

    def test_image_upload_and_filters(self):
        path = Path('data/images/B0001G6U4S.webp')
        with path.open('rb') as source:
            response = self.client.post('/api/search/image', params={'category': 'Digital Cameras', 'max_price': 200},
                                        files={'file': ('camera.webp', source, 'image/webp')})
        self.assertEqual(response.status_code, 200)
        items = response.json()['items']
        self.assertEqual(len(items), 5)
        self.assertTrue(all(p['original_category'] == 'Digital Cameras' and p['price'] <= 200 for p in items))
        invalid = self.client.post('/api/search/image', files={'file': ('bad.png', b'not an image', 'image/png')})
        self.assertEqual(invalid.status_code, 422)
        large = self.client.post('/api/search/image', files={'file': ('large.png', b'x' * (MAX_UPLOAD_BYTES + 1), 'image/png')})
        self.assertEqual(large.status_code, 413)

    def test_recommendations_and_sentiment_reviews(self):
        product_id = 'B0001G6U4S'
        recommendations = self.client.get(f'/api/products/{product_id}/recommendations').json()['items']
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(p['product_id'] != product_id and p['category'] == 'Cameras & Photography' and p['reasons'] for p in recommendations))
        response = self.client.get(f'/api/products/{product_id}/sentiment').json()
        self.assertEqual(response['total_reviews'], response['summary']['review_count'])
        self.assertTrue(5 <= response['total_reviews'] <= 10)
        self.assertTrue(all(review['review_id'] and review['text'] and review['sentiment'] in ('positive', 'negative', 'neutral') for review in response['reviews']))
        for label in ('positive', 'negative', 'neutral'):
            filtered = self.client.get(f'/api/products/{product_id}/sentiment', params={'sentiment': label}).json()
            self.assertEqual(filtered['total_reviews'], response['summary'][f'{label}_count'])
            self.assertTrue(all(review['sentiment'] == label for review in filtered['reviews']))
        self.assertEqual(self.client.get(f'/api/products/{product_id}/sentiment', params={'sentiment': 'mixed'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
