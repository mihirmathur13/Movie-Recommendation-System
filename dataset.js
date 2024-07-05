document.addEventListener('DOMContentLoaded', function () {
    const rowsPerPage = 10;
    let currentPage = 1;
    let dataset = [];

    function fetchDataset() {
        fetch('/api/dataset')
            .then(response => response.json())
            .then(data => {
                dataset = data.movies;
                displayPage(1);
            })
            .catch(error => console.error('Error fetching dataset:', error));
    }

    function displayPage(page) {
        currentPage = page;
        const tableContainer = document.getElementById('dataset-table-container');
        const paginationControls = document.getElementById('pagination-controls');

        // Clear previous content
        tableContainer.innerHTML = '';
        paginationControls.innerHTML = '';

        // Create table
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');

        // Table headers
        const headers = ['Title', 'Genre', 'Director', 'Cast', 'Year', 'Rating'];
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            tr.appendChild(th);
        });
        thead.appendChild(tr);

        // Table rows
        const start = (page - 1) * rowsPerPage;
        const end = start + rowsPerPage;
        const paginatedItems = dataset.slice(start, end);
        paginatedItems.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${item.title}</td>
                <td>${item.genre}</td>
                <td>${item.director}</td>
                <td>${item.cast}</td>
                <td>${item.year}</td>
                <td>${item.rating}</td>
            `;
            tbody.appendChild(tr);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        tableContainer.appendChild(table);

        // Pagination controls
        const totalPages = Math.ceil(dataset.length / rowsPerPage);

        for (let i = 1; i <= totalPages; i++) {
            const button = document.createElement('button');
            button.textContent = i;
            button.disabled = i === currentPage;
            button.addEventListener('click', () => displayPage(i));
            paginationControls.appendChild(button);
        }
    }

    fetchDataset();
});
