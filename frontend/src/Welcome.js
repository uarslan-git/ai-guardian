import { motion } from 'framer-motion';
import './App.css';

function Welcome() {
    return (
        <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <header className="App-header">
                <h1>Ai-Guardian Assignment</h1>
                <h2> A beginners Guide To Underfitting/Overfitting</h2>
                <p>
                    This is a simple React application designed to help you get started with building amazing web applications.
                </p>
            </header>
        </motion.div>
    );
}

export default Welcome;