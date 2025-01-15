import axios from "axios";
import React, { useEffect, useState } from "react";

const PokemonList = () => {
    const [pokemon, setPokemon] = useState([]);
    const [error, setError] = useState(null);  // State to store any errors
    const [loading, setLoading] = useState(true) // State for loading status;

    useEffect(() => {
        // Fetch data from PokeAPI
        const fetchPokemon = async () => {
            try {
                const response = await axios.get("https://pokeapi.co/api/v2/pokemon?limit=10");
                setPokemon(response.data.results); // Store results in state
            } catch (err) {
                setError(err.message); // Handle any errors
            } finally {
                setLoading(false); // Set loading to false when done
            }
        };
        fetchPokemon();
    }, []);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div>
            <h1>Pokemon List</h1>
            <ul>
                {pokemon.map((poke, index) => (
                    <li key={index}>{poke.name}</li>
                ))}
            </ul>
        </div>
    );
};

export default PokemonList;
