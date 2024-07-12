document.addEventListener("DOMContentLoaded", function () {
    const movieList = document.getElementById("movieList");

    // Function to fetch and display movies from JSON
    function fetchAndDisplayMoviesFromJSON() {
      fetch("movies.json")
        .then((response) => response.json())
        .then((data) => {
          data.forEach((movie) => {
            const li = document.createElement("li");
            li.textContent = `${movie.title} - ${movie.genre}`;
            movieList.appendChild(li);
          });
        })
        .catch((error) =>
          console.error("Error fetching movies from JSON:", error)
        );
    }

    // Function to fetch and display movies from CSV (example)
    function fetchAndDisplayMoviesFromCSV() {
      // Example: Fetching movies from CSV using PapaParse (https://www.papaparse.com/)
      Papa.parse("unique_movie.csv", {
        download: true,
        header: true,
        complete: function (results) {
          results.data.forEach((movie) => {
            const li = document.createElement("li");
            li.textContent = `${movie.Title} - ${movie.Genre}`; // Adjust fields based on your CSV structure
            movieList.appendChild(li);
          });
        },
        error: function (error) {
          console.error("Error fetching movies from CSV:", error);
        },
      });
    }

    // Call functions to fetch and display movies when the page loads
    fetchAndDisplayMoviesFromJSON();
    fetchAndDisplayMoviesFromCSV(); // This will fetch and display movies from CSV
  });
  function fetchDataset() {
    fetch("/api/dataset")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        dataset = data.movies || [];
        console.log("Fetched dataset:", dataset); // Log dataset to check if data is fetched
        displayPage(1);
      })
      .catch((error) => console.error("Error fetching dataset:", error.message));
  }
