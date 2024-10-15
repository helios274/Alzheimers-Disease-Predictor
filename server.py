from flask import Flask, request, render_template
from predictor.index import make_prediction

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        data = request.form
        prediction = make_prediction(data)

        return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
