// document.addEventListener("DOMContentLoaded", function() {
//     fetch('/api/movies')
//         .then(response => response.json())
//         .then(movies => {
//             // Set up autocomplete for likedMovies input field
//             $("#likedMovies").autocomplete({
//                 source: movies
//             });

//             // Set up autocomplete for preferredGenres input field
//             $("#preferredGenres").autocomplete({
//                 source: movies
//             });

//             // Set up autocomplete for preferredCastDirector input field
//             $("#preferredCastDirector").autocomplete({
//                 source: movies
//             });

//             // Set up autocomplete for preferredDirector input field
//             $("#preferredDirector").autocomplete({
//                 source: movies
//             });
//         })
//         .catch(error => console.error('Error fetching movie list:', error));
// });

function showForm(formId) {
    const forms = document.querySelectorAll('.form');
    forms.forEach(form => form.style.display = 'none');
    document.getElementById(formId).style.display = 'block';
}

function showTopMovies() {
    fetchRecommendations('/api/top-movies');
}

function submitDemographicForm(event) {
    event.preventDefault();
    const ageInput = document.getElementById('age');
    const age = ageInput.value;
    const gender = document.getElementById('gender').value;
    const state = document.getElementById('state').value;
    const job = document.getElementById('job').value;

    // Check if the age is a valid integer
    if (!Number.isInteger(Number(age)) || age < 10 || age > 84) {
        alert('Please enter a valid age between 10 and 84.');
        return;
    }

    const data = { age: parseInt(age, 10), gender, state, job };
    fetchRecommendations('/api/recommend/demographic', data);
}

function submitLikedMoviesForm(event) {
    event.preventDefault();
    const likedMovies = document.getElementById('likedMovies').value.split(',').map(movie => movie.trim());
    const data = { likedMovies };
    fetchRecommendations('/api/recommend/liked', data);
}``

function submitGenresForm(event) {
    event.preventDefault();
    const preferredGenres = document.getElementById('preferredGenres').value.split(',').map(movie => movie.trim());
    const data = { likedMovies: preferredGenres };
    fetchRecommendations('/api/recommend/genre', data);
}

function submitCastDirectorForm(event) {
    event.preventDefault();
    const preferredCastDirector = document.getElementById('preferredCastDirector').value.split(',').map(movie => movie.trim());
    const data = { likedMovies: preferredCastDirector };
    fetchRecommendations('/api/recommend/cast-director', data);
}
function submitDirectorForm(event) {
    event.preventDefault();
    const preferredDirector = document.getElementById('preferredDirector').value.split(',').map(movie => movie.trim());
    const data = { likedMovies: preferredDirector };
    fetchRecommendations('/api/recommend/director', data);
}

function fetchRecommendations(url, data) {
    const recommendationsContainer = document.getElementById('recommendations');
    const recommendationList = document.getElementById('recommendationList');
    recommendationList.innerHTML = '';

    fetch(`http://127.0.0.1:5000/${url}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.recommendations && Array.isArray(data.recommendations)) {
            data.recommendations.forEach(movie => {
                const listItem = document.createElement('li');
                listItem.textContent = movie.name;
                recommendationList.appendChild(listItem);
            });
        } else {
            const listItem = document.createElement('li');
            listItem.textContent = 'No recommendations found.';
            recommendationList.appendChild(listItem);
        }
        recommendationsContainer.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        recommendationsContainer.style.display = 'block';
        const listItem = document.createElement('li');
        listItem.textContent = 'Error fetching recommendations. Please try again later.';
        recommendationList.appendChild(listItem);
    });
}
