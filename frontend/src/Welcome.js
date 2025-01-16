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
                    Stell dir mal vor du würdest nicht wissen, was ein Hund und eine Katze sind. Nun zeigt man dir ein Beispielbild von beidem.
                    Vermutlich würdest du sofort die Unterschiede erkennen können. Zeigt man dir nun Bilder von anderen Hunden, weißt du schon,
                    dass das keine Katzen sind und könntest dutzende von Gründen aufzählen wieso.

                    Maschinen fällt das schwieriger. Die Regeln werden sich anhand der Beispiele, die man ihr gibt erschlossen.
                    Dabei können die Regeln unterschiedlich allgemein sein.
                    Wenn die Regeln zu speziell sind redet man on Overfitting, wenn sie zu allgemein gehalten sind, redet man von underfitting.

                    In der Realität kannst du dir das so vorstellen:
                    Ein Modell, das overfittet ist, hat anhand der Daten gelernt, dass jedes Bild, wo ein vierbeiniges Tier auf dem Rasen
                    mit einem Tennisball und einer Frisbey zu sehen ist, muss ein Hund sein. Wenn man nun aber ein Bild eines Huskys zeigt, wäre
                    das Modell verwirrt, weil es seine Regeln gar nicht richtig anwenden kann.

                    Umgekehrt können Regeln aber auch viel zu allgemein und damit das Modell underfittet sein.
                    Die Regel könnte hier lauten, dass alles ein Hund ist, was 4 Beine hat.
                    Diese Regel ist offensichtlich nicht sonderlich gut, denn auch Katzen haben meistens
                    4 Beine.
                </p>
            </header>
        </motion.div>
    );
}

export default Welcome;