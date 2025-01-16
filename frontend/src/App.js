import PokemonList from './PokemonList';
import Welcome from './Welcome';
function App() {
  return (
    <div className="App">
      <Welcome />
      <PokemonList />
      <iframe class="Plot" src="http://127.0.0.1:5000/api/get_plot_html/assets/testplot.pkl"></iframe>
    </div>
  );
}

export default App;