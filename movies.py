from flask import Flask, render_template
import csv

app = Flask(__name__)

@app.route('/dataset')
def dataset():
    movie_data = []
    with open('unique_movies.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            movie_data.append(row)

    return render_template('dataset.html', movies=movie_data)

if __name__ == '__main__':
    app.run(debug=True)
