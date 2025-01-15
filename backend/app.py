from flask import Flask
import pickle
import os

app = Flask(__name__)

def root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@app.route('/api/hello')
def hello():
    return {'message': 'hello world from the backend'}

@app.route('/api/get_plot/<path:path>')
def get_plot(path):
    fig = pickle.load(open(os.path.join(root_dir(), path), 'rb'))
    return fig.to_json()

@app.route('/api/get_plot_html/<path:path>')
def get_plot_html(path):
    fig = pickle.load(open(os.path.join(root_dir(), path), 'rb'))
    return fig.to_html()


if __name__ == '__main__':
    app.run(port=5000, debug=True) 