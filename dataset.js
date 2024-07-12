document.addEventListener("DOMContentLoaded", function () {
    const rowsPerPage = 10;
    let currentPage = 1;
    let dataset = [];

    function fetchCSVData() {
      fetch("unique_movie.csv")
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.text();
        })
        .then((csvData) => {
          parseCSV(csvData);
        })
        .catch((error) =>
          console.error("Error fetching CSV data:", error.message)
        );
    }

    function parseCSV(csvData) {
      Papa.parse(csvData, {
        header: true,
        complete: function (results) {
          console.log("Parsed CSV:", results.data);
          dataset = results.data || [];
          displayPage(1);
        },
        error: function (error) {
          console.error("Error parsing CSV:", error);
        },
      });
    }

    function displayPage(page) {
      currentPage = page;
      const movieList = document.getElementById("movieList");

      // Clear previous content
      movieList.innerHTML = "";

      // Table rows
      const start = (page - 1) * rowsPerPage;
      const end = start + rowsPerPage;
      const paginatedItems = dataset.slice(start, end);
      paginatedItems.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item.name; // Displaying only movie name
        movieList.appendChild(li);
      });

      // Implement pagination controls if needed
    }

    // Fetch CSV data on page load
    fetchCSVData();
  });
