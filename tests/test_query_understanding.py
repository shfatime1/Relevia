import unittest

from backend.query_understanding import understand_query
from backend.unified_search import UnifiedSearch


class QueryUnderstandingTests(unittest.TestCase):
    def test_azerbaijani_and_english_intent(self):
        query = understand_query('100 dollardan ucuz simsiz qulaqlıq')
        self.assertEqual(query['text'], 'wireless headphones')
        self.assertEqual(query['product_type'], 'headphones')
        self.assertEqual(query['category'], 'Headphones & Earbuds')
        self.assertLess(query['max_price'], 100)
        query = understand_query('Mənə dərs üçün münasib qiymətə laptop tap')
        self.assertEqual(query['intent'], 'study')
        self.assertIn('affordable', query['text'])
        self.assertIsNone(query['max_price'])
        query = understand_query('Gaming mouse with good ratings')
        self.assertEqual(query['text'], 'Gaming mouse with good ratings')
        self.assertIn('gaming', query['attributes'])
        self.assertIsNone(query['min_rating'])

    def test_explicit_constraints_and_safe_fallback(self):
        query = understand_query('Find laptops under $500')
        self.assertEqual(query['text'], 'find laptops')
        self.assertLess(query['max_price'], 500)
        self.assertEqual(understand_query('mouse 4+ stars')['min_rating'], 4)
        self.assertEqual(understand_query('mouse minimum rating 4.5')['min_rating'], 4.5)
        self.assertEqual(understand_query('mouse at least 4.5/5 stars')['min_rating'], 4.5)
        self.assertEqual(understand_query('siçan 4 ulduzdan yuxarı')['min_rating'], 4)
        self.assertIsNone(understand_query('mouse under 4 stars')['min_rating'])
        self.assertIsNone(understand_query('mouse under 4 stars')['max_price'])
        self.assertIsNone(understand_query('mouse under 50 manat')['max_price'])
        self.assertIsNone(understand_query('laptop with at least 16 GB')['min_price'])
        self.assertIsNone(understand_query('laptop at least 4')['min_price'])
        self.assertEqual(understand_query('wireless headphones')['text'], 'wireless headphones')

    def test_filters_skip_missing_rating(self):
        products = [dict(parent_asin=str(i), title='mouse', price=price, average_rating=rating,
                         main_category='Computers & Components', original_category='Computer Accessories & Peripherals',
                         image_path='image.webp')
                    for i, (price, rating) in enumerate(((20, 4.5), (25, 3.5), (30, None)))]
        ranked = [{'product': product, 'score': 1, 'reasons': []} for product in products]
        results = UnifiedSearch.filter_results(ranked, min_price=20, max_price=30, min_rating=4)
        self.assertEqual([row['product_id'] for row in results], ['0'])


if __name__ == '__main__':
    unittest.main()
