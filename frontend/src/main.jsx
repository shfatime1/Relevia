import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowDownRight, ArrowLeft, ArrowRight, AudioLines, Camera, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Headphones, Laptop, Menu, Monitor, Package, Radio, Search, ShieldCheck, SlidersHorizontal, Sparkles, Star, Tv, Upload, Watch, Wifi, X, Zap } from 'lucide-react';
import './style.css';

const PAGE_SIZE = 12;
const icons = {
  'Cameras & Photography': Camera, 'Computers & Components': Laptop,
  'Data Storage': Package, 'Headphones & Earbuds': Headphones, 'Networking': Wifi,
  'Portable Audio & Radios': Radio, 'Power & Lighting': Zap,
  'Security & Surveillance': ShieldCheck, 'Speakers & Home Audio': AudioLines,
  'TV & Video Players': Tv, 'Vehicle Electronics & GPS': Monitor, 'Wearable Technology': Watch,
};

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach the backend. Start FastAPI on 127.0.0.1:8000, then try again.');
  }
  if (!response.ok) {
    let message = response.status >= 500
      ? 'The backend is unavailable or returned an error. Check FastAPI on 127.0.0.1:8000, then try again.'
      : 'Something went wrong. Please try again.';
    try { const body = await response.json(); if (typeof body.detail === 'string') message = body.detail; } catch { /* use the default */ }
    throw new Error(message);
  }
  try {
    return await response.json();
  } catch {
    throw new Error('The server returned an unexpected response. Check that FastAPI is running on 127.0.0.1:8000.');
  }
}

function params(values) {
  return new URLSearchParams(Object.entries(values).filter(([, value]) => value !== '' && value != null)).toString();
}

const money = value => typeof value === 'number' ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value) : 'Price unavailable';
const productIdFromHash = () => decodeURIComponent(window.location.hash.match(/^#\/product\/([^/]+)$/)?.[1] || '');

function Logo() {
  return <a className="brand" href="#home" aria-label="Relevia home"><span className="brand-mark"><Sparkles size={18} strokeWidth={2.4} /></span><span>relevia<span className="brand-dot">.</span></span></a>;
}

function Rating({ value, count }) {
  return <span className="rating"><Star size={14} fill="currentColor" strokeWidth={1.5} /> {value != null ? Number(value).toFixed(1) : '—'} {count != null && <span className="rating-count">({Number(count).toLocaleString()})</span>}</span>;
}

function ProductCard({ item, catalog }) {
  const product = { ...catalog[item.product_id], ...item };
  return <a className="product-card" href={`#/product/${encodeURIComponent(product.product_id)}`}>
    <div className="card-image"><img src={product.image_url} alt="" loading="lazy" /><span className="card-arrow"><ArrowUpRightIcon /></span></div>
    <div className="card-copy"><p className="card-category">{product.original_category || product.category}</p><h3>{product.title}</h3><div className="card-bottom"><strong>{money(product.price)}</strong><Rating value={product.rating} count={product.rating_count} /></div></div>
  </a>;
}

function ArrowUpRightIcon() { return <ArrowRight size={17} className="arrow-diagonal" />; }

function App() {
  const [productId, setProductId] = useState(productIdFromHash);
  const [categories, setCategories] = useState([]);
  const [catalog, setCatalog] = useState({});
  const [category, setCategory] = useState('');
  const [priceDraft, setPriceDraft] = useState({ min: '', max: '' });
  const [prices, setPrices] = useState({ min: '', max: '' });
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState(null);
  const [page, setPage] = useState(1);
  const [data, setData] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [metadataError, setMetadataError] = useState('');
  const [metadataVersion, setMetadataVersion] = useState(0);
  const [filterError, setFilterError] = useState('');
  const [mobileMenu, setMobileMenu] = useState(false);
  const [readiness, setReadiness] = useState('preparing');
  const [readinessAttempt, setReadinessAttempt] = useState(0);
  const fileRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    let timer;
    let controller;
    const started = Date.now();
    setReadiness('preparing');
    async function check() {
      controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        if (import.meta.env.DEV) {
          const probe = await fetch('/__backend_ready', { signal: controller.signal, cache: 'no-store' });
          if (!probe.ok || !(await probe.json()).ready) throw new Error('Backend is starting');
        }
        const data = await api('/api/categories', { signal: controller.signal });
        if (!cancelled) { setCategories(data.main_categories); setReadiness('ready'); }
      } catch {
        if (cancelled) return;
        if (Date.now() - started >= 180000) setReadiness('unavailable');
        else timer = setTimeout(check, 2000);
      } finally {
        clearTimeout(timeout);
      }
    }
    check();
    return () => { cancelled = true; clearTimeout(timer); controller?.abort(); };
  }, [readinessAttempt]);

  useEffect(() => {
    const onHash = () => {
      const id = productIdFromHash();
      setProductId(id);
      if (id || window.location.hash === '#home') window.scrollTo({ top: 0, behavior: 'instant' });
      else if (window.location.hash === '#catalog' || window.location.hash === '#about') {
        setTimeout(() => document.getElementById(window.location.hash.slice(1))?.scrollIntoView({ behavior: 'smooth' }), 30);
      }
    };
    window.addEventListener('hashchange', onHash); return () => window.removeEventListener('hashchange', onHash);
  }, []);
  useEffect(() => {
    if (readiness !== 'ready') return;
    const controller = new AbortController();
    setMetadataError('');
    api('/api/categories', { signal: controller.signal }).then(d => setCategories(d.main_categories)).catch(e => { if (e.name !== 'AbortError') setMetadataError('Categories or product ratings could not be loaded.'); });
    // Catalog cards include ratings; search and recommendation responses do not.
    api('/api/products?page=1&page_size=100', { signal: controller.signal }).then(async first => {
      const rest = await Promise.all(Array.from({ length: Math.ceil(first.total / 100) - 1 }, (_, i) => api(`/api/products?page=${i + 2}&page_size=100`, { signal: controller.signal })));
      setCatalog(Object.fromEntries([first, ...rest].flatMap(result => result.items).map(item => [item.product_id, item])));
    }).catch(e => { if (e.name !== 'AbortError') setMetadataError('Categories or product ratings could not be loaded.'); });
    return () => controller.abort();
  }, [metadataVersion, readiness]);
  useEffect(() => {
    if (productId || readiness !== 'ready') return;
    const controller = new AbortController();
    const options = { category, min_price: prices.min, max_price: prices.max };
    let path, request = { signal: controller.signal };
    if (search?.type === 'text') path = `/api/search/text?${params({ q: search.query, ...options, limit: 100 })}`;
    else if (search?.type === 'image') { path = `/api/search/image?${params({ ...options, limit: 100 })}`; const form = new FormData(); form.append('file', search.file); request = { ...request, method: 'POST', body: form }; }
    else path = `/api/products?${params({ ...options, page, page_size: PAGE_SIZE })}`;
    setLoading(true); setError('');
    api(path, request).then(result => { if (!controller.signal.aborted) setData({ items: result.items, total: search ? result.items.length : result.total }); }).catch(e => { if (e.name !== 'AbortError') setError(e.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [productId, category, prices, search, search ? 0 : page, readiness]);

  const submitSearch = event => { event.preventDefault(); if (readiness !== 'ready' || !query.trim()) return; setPage(1); setSearch({ type: 'text', query: query.trim() }); window.location.hash = 'catalog'; setTimeout(() => document.getElementById('catalog')?.scrollIntoView({ behavior: 'smooth' }), 30); };
  const upload = event => { const file = event.target.files?.[0]; event.target.value = ''; if (!file) return; if (file.size > 10 * 1024 * 1024) { setError('Please choose an image smaller than 10 MiB.'); return; } setQuery(''); setPage(1); setSearch({ type: 'image', file }); window.location.hash = 'catalog'; setTimeout(() => document.getElementById('catalog')?.scrollIntoView({ behavior: 'smooth' }), 30); };
  const resetSearch = () => { setSearch(null); setQuery(''); setPage(1); };
  const retry = () => { setError(''); setMetadataVersion(value => value + 1); setPrices({ ...prices }); };
  const selectCategory = value => { setCategory(value); setPage(1); setMobileMenu(false); window.location.hash = 'catalog'; };
  const applyPrices = event => { event.preventDefault(); const min = priceDraft.min === '' ? null : Number(priceDraft.min); const max = priceDraft.max === '' ? null : Number(priceDraft.max); if ((min != null && (!Number.isFinite(min) || min < 0)) || (max != null && (!Number.isFinite(max) || max < 0)) || (min != null && max != null && min > max)) { setFilterError('Enter a valid price range.'); return; } setFilterError(''); setPrices({ ...priceDraft }); setPage(1); };
  const resetFilters = () => { setCategory(''); setPriceDraft({ min: '', max: '' }); setPrices({ min: '', max: '' }); setFilterError(''); setPage(1); };
  const heroProduct = Object.values(catalog).find(p => p.product_id === 'B0BNC1QBH6') || Object.values(catalog).find(p => p.category === 'Headphones & Earbuds');
  const items = search ? data.items.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE) : data.items;
  const pages = Math.ceil(data.total / PAGE_SIZE);

  return <>
    <div className="announcement"><span>SMARTER SEARCH, BETTER FINDS</span><span className="announcement-right">A little clarity for your next big find <Sparkles size={13} /></span></div>
    <header className="header"><div className="header-inner"><Logo /><nav className="desktop-nav"><a href="#catalog">Explore products</a><a href="#about">About Relevia</a></nav><div className="header-end"><span className="header-caption">DISCOVER SOMETHING GOOD</span><a href="#catalog" className="header-link">Start exploring <ArrowRight size={15} /></a></div><button className="mobile-menu-btn icon-button" aria-label="Toggle navigation" aria-expanded={mobileMenu} onClick={() => setMobileMenu(!mobileMenu)}>{mobileMenu ? <X /> : <Menu />}</button></div>{mobileMenu && <nav className="mobile-nav"><a href="#catalog" onClick={() => setMobileMenu(false)}>Explore products</a><a href="#about" onClick={() => setMobileMenu(false)}>About Relevia</a></nav>}</header>
    <main>
      {productId ? readiness === 'ready' ? <ProductDetail id={productId} catalog={catalog} /> : <div className="detail-state" role="status">{readiness === 'preparing' ? 'Preparing search…' : <>The backend is unavailable. <button onClick={() => setReadinessAttempt(value => value + 1)}>Try again</button></>}</div> : <>
        <section className="hero wrap"><div className="hero-copy"><div className="eyebrow"><span className="spark">✳</span> DISCOVER TECH THAT FITS</div><h1>Find your<br /><em>kind of tech.</em></h1><p>Less searching. More finding. Explore electronics that feel just right for you.</p><div className="hero-actions"><a className="button button-dark" href="#catalog">Explore products <ArrowRight size={17} /></a><span className="hero-note"><span className="mini-spark">✳</span> A smarter way to discover</span></div><div className="hero-index"><span>01 / THE DISCOVERY EDIT</span><span>MADE FOR CURIOUS MINDS</span></div></div><div className="hero-visual"><div className="hero-orbit orbit-one"></div><div className="hero-orbit orbit-two"></div><span className="hero-star">✳</span><span className="hero-image-label">A LITTLE SOMETHING<br />FOR EVERY DAY</span>{heroProduct && <a className="hero-product" href={`#/product/${heroProduct.product_id}`}><img src={heroProduct.image_url} alt={heroProduct.title} /></a>}<div className="hero-sticker"><span>CURIOUS<br />BY NATURE</span><ArrowDownRight size={21} /></div><div className="hero-product-tag"><span className="tag-dot"></span> {heroProduct?.original_category || 'Thoughtful discoveries'}</div><span className="hero-number">RE / 001</span></div></section>
        <section className="search-section wrap" aria-label="Search products"><div className="search-intro"><span className="eyebrow muted">LET'S FIND IT</span><h2>What are you looking for?</h2><p>Search by a few words or a photo. We’ll take it from there.</p></div>{readiness === 'preparing' && <div className="inline-loading search-readiness" role="status"><span className="spinner" /> Preparing search… This may take a minute while the models load.</div>}{readiness === 'unavailable' && <div className="state-message error-message" role="alert"><strong>Search is unavailable.</strong><span>Check that the backend is running, then try again.</span><button onClick={() => setReadinessAttempt(value => value + 1)}>Try again</button></div>}<form className="search-box" onSubmit={submitSearch}><Search size={21} /><input value={query} onChange={e => setQuery(e.target.value)} aria-label="Search products" placeholder="Try “wireless headphones” or “kamera”" /><input type="file" ref={fileRef} accept="image/*" onChange={upload} className="sr-only" aria-label="Choose an image to search" /><button type="button" disabled={readiness !== 'ready'} className="upload-button" onClick={() => fileRef.current?.click()}><Camera size={19} /><span>Search by image</span></button><button type="submit" disabled={readiness !== 'ready'} className="search-submit" aria-label="Submit search"><ArrowRight size={20} /></button></form><div className="search-caption"><span><Sparkles size={13} /> SEARCH BY WHAT MATTERS TO YOU</span><span>ENGLISH & AZƏRBAYCANCA</span></div></section>
        <section className="catalog-section wrap" id="catalog"><div className="section-heading"><div><div className="eyebrow muted"><span className="spark">✳</span> THE GOOD STUFF</div><h2>Discover your next favorite<span className="heading-dot">.</span></h2><p>Good finds for everyday life, all in one place.</p></div><span className="catalog-counter">{!loading && !search ? `${data.total.toLocaleString()} FINDS` : 'CURATED FOR YOU'} <ArrowDownRight size={17} /></span></div>
          <div className="category-strip" aria-label="Browse categories"><button className={`category-chip ${!category ? 'active' : ''}`} onClick={() => selectCategory('')}><span className="chip-icon"><Sparkles size={17} /></span>All products</button>{categories.map(name => { const Icon = icons[name] || Package; return <button className={`category-chip ${category === name ? 'active' : ''}`} key={name} onClick={() => selectCategory(name)}><span className="chip-icon"><Icon size={17} /></span>{name}</button>; })}</div>{metadataError && !error && <p className="field-error" role="status">{metadataError} <button className="text-button" onClick={retry}>Try again</button></p>}
          <div className="catalog-layout"><aside className="filters"><div className="filter-top"><h3><SlidersHorizontal size={17} /> Filters</h3><button className="text-button" onClick={resetFilters}>Clear all</button></div><div className="filter-block"><h4>Category <ChevronDown size={16} /></h4><div className="category-list"><button className={!category ? 'selected' : ''} onClick={() => selectCategory('')}><span>All categories</span>{!category && <Check size={15} />}</button>{categories.map(name => <button key={name} className={category === name ? 'selected' : ''} onClick={() => selectCategory(name)}><span>{name}</span>{category === name && <Check size={15} />}</button>)}</div></div><form className="filter-block" onSubmit={applyPrices}><h4>Price range <ChevronDown size={16} /></h4><div className="price-inputs"><label><span>MIN</span><div>$ <input type="number" min="0" step="0.01" placeholder="0" value={priceDraft.min} onChange={e => setPriceDraft({ ...priceDraft, min: e.target.value })} /></div></label><span className="price-dash">—</span><label><span>MAX</span><div>$ <input type="number" min="0" step="0.01" placeholder="Any" value={priceDraft.max} onChange={e => setPriceDraft({ ...priceDraft, max: e.target.value })} /></div></label></div>{filterError && <p className="field-error" role="alert">{filterError}</p>}<button className="apply-button" type="submit">Apply price</button></form><div className="filter-help"><CircleHelp size={17} /><p>Looking for something specific?<br /><strong>Try a photo search above.</strong></p></div></aside>
            <div className="results"><div className="results-bar"><div><strong>{search ? (search.type === 'text' ? `Results for “${search.query}”` : 'Visual matches') : category || 'All products'}</strong><span>{readiness === 'unavailable' ? 'Backend unavailable' : readiness === 'preparing' ? 'Preparing search…' : loading ? 'Finding your products…' : error ? 'Results unavailable' : `${data.total.toLocaleString()} ${search && data.total === 100 ? 'top matches' : 'products'}`}</span></div>{search && <button className="clear-search" onClick={resetSearch}><X size={14} /> Clear search</button>}<span className="results-label">EXPLORE THE COLLECTION</span></div>{error && <div className="state-message error-message" role="alert"><strong>We couldn’t load these products.</strong><span>{error}</span><button onClick={retry}>Try again</button></div>}{readiness !== 'unavailable' && !error && loading && <div className="product-grid" aria-label="Loading products">{Array.from({ length: 8 }, (_, i) => <div className="skeleton-card" key={i}><div /><span /><span /></div>)}</div>}{!error && !loading && !items.length && <div className="state-message"><Search size={28} /><strong>No products found</strong><span>Try another search or adjust your filters.</span><button onClick={() => { resetFilters(); resetSearch(); }}>Clear filters</button></div>}{!error && !loading && items.length > 0 && <><div className="product-grid">{items.map(item => <ProductCard key={item.product_id} item={item} catalog={catalog} />)}</div>{pages > 1 && <div className="pagination"><button disabled={page === 1} onClick={() => { setPage(page - 1); document.getElementById('catalog')?.scrollIntoView(); }}><ChevronLeft size={16} /> Previous</button><span>Page {page} of {pages}</span><button disabled={page >= pages} onClick={() => { setPage(page + 1); document.getElementById('catalog')?.scrollIntoView(); }}>Next <ChevronRight size={16} /></button></div>}</>}</div>
          </div></section>
      </>}
    </main><footer id="about"><div className="footer-inner wrap"><div><Logo /><p>Good tech, found simply.<br />A little more discovery, a lot less noise.</p></div><div className="footer-note"><span>MADE FOR CURIOUS MINDS <span>✳</span></span><p>Explore real electronics and real customer reviews. Sentiment is model-predicted and may be imperfect.</p></div><div className="footer-links"><a href="#catalog">Explore products</a><a href="#home">Back to top ↑</a></div></div><div className="footer-bottom wrap"><span>© RELEVIA</span><span>THOUGHTFULLY FOUND, JUST FOR YOU.</span></div></footer>
  </>;
}

function ProductDetail({ id, catalog }) {
  const [product, setProduct] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [sentiment, setSentiment] = useState(null);
  const [reviewFilter, setReviewFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [reviewLoading, setReviewLoading] = useState(true);
  const [recommendationLoading, setRecommendationLoading] = useState(true);
  const [error, setError] = useState('');
  const [recommendationError, setRecommendationError] = useState('');
  const [reviewError, setReviewError] = useState('');
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setProduct(null); setRecommendations([]); setSentiment(null); setReviewFilter(''); setError(''); setRecommendationError(''); setRecommendationLoading(true);
    api(`/api/products/${encodeURIComponent(id)}`, { signal: controller.signal }).then(setProduct).catch(e => { if (e.name !== 'AbortError') setError(e.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    api(`/api/products/${encodeURIComponent(id)}/recommendations`, { signal: controller.signal }).then(d => setRecommendations(d.items)).catch(e => { if (e.name !== 'AbortError') setRecommendationError('Similar products could not be loaded.'); }).finally(() => { if (!controller.signal.aborted) setRecommendationLoading(false); });
    return () => controller.abort();
  }, [id]);
  useEffect(() => {
    const controller = new AbortController(); setReviewLoading(true); setReviewError('');
    api(`/api/products/${encodeURIComponent(id)}/sentiment?${params({ sentiment: reviewFilter })}`, { signal: controller.signal }).then(setSentiment).catch(e => { if (e.name !== 'AbortError') setReviewError('Reviews could not be loaded.'); }).finally(() => { if (!controller.signal.aborted) setReviewLoading(false); });
    return () => controller.abort();
  }, [id, reviewFilter]);
  if (loading) return <div className="detail-state wrap"><span className="spinner" /> Finding your product…</div>;
  if (error || !product) return <div className="detail-state wrap"><strong>We couldn’t find this product.</strong><p>{error}</p><a href="#catalog">Back to products <ArrowRight size={16} /></a></div>;
  const s = sentiment?.summary;
  return <div className="detail wrap"><a href="#catalog" className="back-link"><ArrowLeft size={16} /> Back to all products</a><div className="breadcrumbs">HOME <span>/</span> {product.category.toUpperCase()} <span>/</span> PRODUCT DETAILS</div><section className="detail-main"><div className="detail-image"><span className="detail-image-label">RELEVIA / THE GOOD STUFF</span><img src={product.image_url} alt={product.title} /><span className="detail-image-spark">✳</span></div><div className="detail-info"><span className="eyebrow muted"><span className="spark">✳</span> {product.original_category.toUpperCase()}</span><h1>{product.title}</h1><div className="detail-rating"><Rating value={product.rating} count={product.rating_count} /><span>Customer rating</span></div><div className="detail-divider" /><span className="detail-price-label">PRICE</span><div className="detail-price">{money(product.price)}</div><p className="detail-brand">{product.brand ? `By ${product.brand}` : 'Selected for your discovery'}</p><div className="detail-callout"><Sparkles size={19} /><div><strong>A little more detail.</strong><span>Get to know this find before you decide.</span></div></div><a href="#reviews" className="button button-dark detail-action">See customer reviews <ArrowDownRight size={17} /></a><p className="detail-disclaimer">Prices and product information are from the source dataset and may not be current.</p></div></section>
    <section className="detail-content"><div className="detail-copy"><div className="eyebrow muted">THE DETAILS</div><h2>Good to know<span className="heading-dot">.</span></h2><h3>About this product</h3>{product.description.length ? product.description.map((text, i) => <p key={i}>{text}</p>) : <p>No description available for this product.</p>}<h3>Features & specifications</h3>{product.specifications.length ? <ul>{product.specifications.map((text, i) => <li key={i}>{text}</li>)}</ul> : <p>No specifications available.</p>}</div><aside className="detail-side-note"><span>✳</span><p>THE LITTLE<br />THINGS MATTER.</p><small>Find something that feels right for your everyday.</small></aside></section>
    <section className="reviews-section" id="reviews"><div className="section-heading"><div><div className="eyebrow muted"><span className="spark">✳</span> REAL PEOPLE, REAL THOUGHTS</div><h2>The word on this one<span className="heading-dot">.</span></h2><p>A closer look at what customers are saying.</p></div></div>{reviewError && <div className="state-message error-message" role="alert">{reviewError}</div>}{!s && reviewLoading && <div className="inline-loading"><span className="spinner" /> Loading reviews…</div>}{s && <div className="reviews-layout"><div className="sentiment-card"><span className="sentiment-icon"><Sparkles size={19} /></span><h3>Customer sentiment</h3><p>A snapshot of {s.review_count} selected customer reviews.</p>{['positive', 'neutral', 'negative'].map(label => <div className="sentiment-row" key={label}><div><span>{label}</span><strong>{s[`${label}_percent`].toFixed(0)}%</strong></div><div className="sentiment-track"><div className={`sentiment-fill ${label}`} style={{ width: `${s[`${label}_percent`]}%` }} /></div></div>)}<small>Sentiment is predicted from review text and may be imperfect.</small></div><div className="review-list"><div className="review-top"><h3>Customer reviews <span>({s.review_count})</span></h3><label>Show <select value={reviewFilter} onChange={e => setReviewFilter(e.target.value)}><option value="">All reviews</option><option value="positive">Positive</option><option value="neutral">Neutral</option><option value="negative">Negative</option></select></label></div>{reviewLoading ? <div className="inline-loading"><span className="spinner" /> Loading reviews…</div> : sentiment.reviews.length ? sentiment.reviews.map(review => <article className="review" key={review.review_id}><div className="review-heading"><div><div className="review-stars" aria-label={`${review.rating} out of 5 stars`}>{'★'.repeat(Math.max(0, Math.min(5, Math.round(review.rating))))}{'☆'.repeat(5 - Math.max(0, Math.min(5, Math.round(review.rating))))}</div><h4>{review.title || 'Customer review'}</h4></div><span className={`sentiment-badge ${review.sentiment}`}>{review.sentiment}</span></div><p>{review.text}</p><div className="review-meta">{review.timestamp ? new Date(review.timestamp).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' }) : ''}{review.verified_purchase && <span>✓ Verified purchase</span>}</div></article>) : <p className="empty-reviews">No {reviewFilter} reviews for this product.</p>}</div></div>}</section>
    <section className="similar-section"><div className="section-heading"><div><div className="eyebrow muted"><span className="spark">✳</span> KEEP DISCOVERING</div><h2>You might also like<span className="heading-dot">.</span></h2><p>A few more finds with a similar feel.</p></div><a className="browse-link" href="#catalog">View all products <ArrowRight size={16} /></a></div>{recommendationLoading ? <div className="inline-loading"><span className="spinner" /> Finding similar products…</div> : recommendationError ? <p>{recommendationError}</p> : recommendations.length ? <div className="product-grid similar-grid">{recommendations.map(item => <ProductCard key={item.product_id} item={item} catalog={catalog} />)}</div> : <p className="muted-text">No similar products available.</p>}</section>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
