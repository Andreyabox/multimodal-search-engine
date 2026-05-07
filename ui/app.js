document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // Search Form Logic
    const searchForm = document.getElementById('search-form');
    const searchBtn = document.getElementById('search-btn');
    const btnText = searchBtn.querySelector('.btn-text');
    const loader = searchBtn.querySelector('.loader');
    
    const resultsContainer = document.querySelector('.results-container');
    const resultsTitle = document.getElementById('results-title');
    const resultsQuery = document.getElementById('results-query');
    const resultsGrid = document.getElementById('results-grid');
    const errorMsg = document.getElementById('results-error');
    const noResults = document.getElementById('no-results');

    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = document.getElementById('query').value.trim();
        const top_k = parseInt(document.getElementById('top_k').value);
        let api_url = document.getElementById('api_url').value.trim();
        
        if (!query) {
            showError("Введите текстовый запрос для поиска.");
            return;
        }

        // Sanitize API URL
        if(api_url.endsWith('/')) {
            api_url = api_url.slice(0, -1);
        }

        // UI state: loading
        searchBtn.disabled = true;
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
        hideError();
        resultsTitle.style.display = 'none';
        resultsGrid.innerHTML = '';
        noResults.style.display = 'none';

        try {
            const response = await fetch(`${api_url}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query, top_k })
            });

            if (!response.ok) {
                let errorDetails = response.statusText;
                try {
                    const errorText = await response.text();
                    if(errorText) errorDetails = errorText;
                } catch(e) {}
                throw new Error(`API вернуло ошибку HTTP ${response.status}: ${errorDetails}`);
            }

            const data = await response.json();
            displayResults(data.results || [], data.query || query);

        } catch (error) {
            showError(`Ошибка поиска: ${error.message}`);
        } finally {
            // UI state: restore
            searchBtn.disabled = false;
            btnText.style.display = 'inline-block';
            loader.style.display = 'none';
        }
    });

    function displayResults(results, queryStr) {
        resultsQuery.textContent = queryStr;
        resultsTitle.style.display = 'block';

        if (results.length === 0) {
            noResults.style.display = 'block';
            return;
        }

        results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'image-card';

            const captionStr = item.caption || 'Без описания';
            const scoreStr = item.score !== undefined ? parseFloat(item.score).toFixed(4) : 'N/A';

            const imgWrapper = document.createElement('div');
            imgWrapper.className = 'img-wrapper';

            if (item.image_url) {
                const image = document.createElement('img');
                image.src = item.image_url;
                image.alt = captionStr;
                image.loading = 'lazy';
                imgWrapper.appendChild(image);
            } else {
                imgWrapper.style.background = 'rgba(255,255,255,0.05)';
                const placeholder = document.createElement('span');
                placeholder.style.color = 'var(--text-secondary)';
                placeholder.textContent = 'Нет изображения';
                imgWrapper.appendChild(placeholder);
            }

            const imgInfo = document.createElement('div');
            imgInfo.className = 'img-info';

            const caption = document.createElement('div');
            caption.className = 'img-caption';
            caption.title = captionStr;
            caption.textContent = captionStr;

            const score = document.createElement('div');
            score.className = 'img-score';
            score.textContent = `Score: ${scoreStr}`;

            imgInfo.append(caption, score);
            card.append(imgWrapper, imgInfo);
            resultsGrid.appendChild(card);
        });
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.style.display = 'block';
    }

    function hideError() {
        errorMsg.style.display = 'none';
    }

    // ---------- Video Search ----------
    const videoForm = document.getElementById('video-search-form');
    const videoBtn = document.getElementById('video-search-btn');
    const videoBtnText = videoBtn.querySelector('.btn-text');
    const videoLoader = videoBtn.querySelector('.loader');
    const videoResultsTitle = document.getElementById('video-results-title');
    const videoResultsQuery = document.getElementById('video-results-query');
    const videoResultsGrid = document.getElementById('video-results-grid');
    const videoErrorMsg = document.getElementById('video-results-error');
    const videoNoResults = document.getElementById('video-no-results');

    videoForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = document.getElementById('video-query').value.trim();
        const top_k = parseInt(document.getElementById('video-top-k').value);
        const mode = document.getElementById('video-mode').value;
        let api_url = document.getElementById('video-api-url').value.trim();

        if (!query) {
            showVideoError("Введите текстовый запрос для поиска.");
            return;
        }
        if (api_url.endsWith('/')) {
            api_url = api_url.slice(0, -1);
        }

        videoBtn.disabled = true;
        videoBtnText.style.display = 'none';
        videoLoader.style.display = 'inline-block';
        hideVideoError();
        videoResultsTitle.style.display = 'none';
        videoResultsGrid.innerHTML = '';
        videoNoResults.style.display = 'none';

        try {
            const response = await fetch(`${api_url}/search/video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k, mode })
            });

            if (!response.ok) {
                let errorDetails = response.statusText;
                try {
                    const errorText = await response.text();
                    if (errorText) errorDetails = errorText;
                } catch (e) {}
                throw new Error(`API вернуло ошибку HTTP ${response.status}: ${errorDetails}`);
            }

            const data = await response.json();
            displayVideoResults(data.results || [], data.query || query, api_url);

        } catch (error) {
            showVideoError(`Ошибка поиска видео: ${error.message}`);
        } finally {
            videoBtn.disabled = false;
            videoBtnText.style.display = 'inline-block';
            videoLoader.style.display = 'none';
        }
    });

    function displayVideoResults(results, queryStr, apiUrl) {
        videoResultsQuery.textContent = queryStr;
        videoResultsTitle.style.display = 'block';

        if (results.length === 0) {
            videoNoResults.style.display = 'block';
            return;
        }

        results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'image-card video-card';

            const title = item.title || item.video_id || 'Без названия';
            const captionStr = item.caption || '';
            const scoreStr = item.score !== undefined ? parseFloat(item.score).toFixed(4) : 'N/A';
            const videoSrc = item.video_url ? `${apiUrl}${item.video_url}` : null;

            const wrapper = document.createElement('div');
            wrapper.className = 'img-wrapper video-wrapper';

            if (videoSrc) {
                const video = document.createElement('video');
                video.controls = true;
                video.preload = 'metadata';
                if (item.thumbnail_url) {
                    video.poster = item.thumbnail_url;
                }
                const source = document.createElement('source');
                source.src = videoSrc;
                source.type = 'video/mp4';
                video.appendChild(source);
                wrapper.appendChild(video);
            } else if (item.thumbnail_url) {
                const image = document.createElement('img');
                image.src = item.thumbnail_url;
                image.alt = title;
                image.loading = 'lazy';
                wrapper.appendChild(image);
            } else {
                const placeholder = document.createElement('span');
                placeholder.style.color = 'var(--text-secondary)';
                placeholder.textContent = 'Нет превью';
                wrapper.appendChild(placeholder);
            }

            const info = document.createElement('div');
            info.className = 'img-info';

            const titleEl = document.createElement('div');
            titleEl.className = 'img-caption video-title';
            titleEl.title = title;
            titleEl.textContent = title;

            const captionEl = document.createElement('div');
            captionEl.className = 'video-caption';
            captionEl.title = captionStr;
            captionEl.textContent = captionStr;

            const meta = document.createElement('div');
            meta.className = 'video-meta';

            const score = document.createElement('span');
            score.className = 'img-score';
            score.textContent = `Score: ${scoreStr}`;
            meta.appendChild(score);

            if (item.youtube_url) {
                const link = document.createElement('a');
                link.className = 'video-yt-link';
                link.href = item.youtube_url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = 'YouTube';
                meta.appendChild(link);
            }

            info.append(titleEl);
            if (captionStr) info.append(captionEl);
            info.append(meta);
            card.append(wrapper, info);
            videoResultsGrid.appendChild(card);
        });
    }

    function showVideoError(msg) {
        videoErrorMsg.textContent = msg;
        videoErrorMsg.style.display = 'block';
    }

    function hideVideoError() {
        videoErrorMsg.style.display = 'none';
    }
});
