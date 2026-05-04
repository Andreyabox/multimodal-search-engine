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
            
            if (item.image_url) {
                card.innerHTML = `
                    <div class="img-wrapper">
                        <img src="${item.image_url}" alt="${captionStr}" loading="lazy" />
                    </div>
                    <div class="img-info">
                        <div class="img-caption" title="${captionStr}">${captionStr}</div>
                        <div class="img-score">Score: ${scoreStr}</div>
                    </div>
                `;
            } else {
                card.innerHTML = `
                    <div class="img-wrapper" style="background: rgba(255,255,255,0.05);">
                        <span style="color: var(--text-secondary);">Нет изображения</span>
                    </div>
                    <div class="img-info">
                        <div class="img-caption" title="${captionStr}">${captionStr}</div>
                        <div class="img-score">Score: ${scoreStr}</div>
                    </div>
                `;
            }
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
});
