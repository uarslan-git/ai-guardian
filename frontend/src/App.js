import { Container, Grid, Typography } from '@mui/material';
import { motion } from 'framer-motion';
import React from 'react';

const App = () => {

  return (
    <div
      style={{
        background: 'linear-gradient(135deg, #f0f4f8, #a6c1e0)',
        display: 'flex',
        padding: '30px 32px', // Add padding to prevent overlap with browser's bookmarks
        minHeight: '100vh', // Ensure the content fills the entire viewport
        flexDirection: 'column',
        justifyContent: 'center',
      }}
    >
      <Container maxWidth="md">
        <Grid container justifyContent="center" alignItems="center" direction="column">
          <Grid item>
            <motion.div
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Typography variant="h3" align="center" color="primary">
                AI-Guardians Assignment
              </Typography>
            </motion.div>
          </Grid>

          <Grid item sx={{ p: 2 }}>
            <motion.div
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <Typography variant="h5" align="center" color="secondary" gutterBottom>
                Explaining Underfitting/Overfitting - for idiots
              </Typography>
            </motion.div>
          </Grid>

          <Grid item>
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 1 }}
            >
              <Typography variant="body1" align="center" color="textSecondary">
                Maschinen lernen etwas anders als wir. Im Gegensatz zu uns brauchen Maschinen viele Mengen Beispiele und lernen anhand dieser Regeln.
                Stell dir mal vor du würdest nicht wissen, was ein Hund und eine Katze sind. Nun zeigt man dir ein Beispielbild von beidem.
                Vermutlich würdest du sofort die Unterschiede erkennen können. Zeigt man dir nun Bilder von anderen Hunden, weißt du schon,
                dass das keine Katzen sind und könntest dutzende von Gründen aufzählen wieso.

                Maschinen fällt das schwieriger. Die Regeln werden sich anhand der Beispiele, die man ihr gibt erschlossen.
                Dabei können die Regeln unterschiedlich allgemein sein.
                Wenn die Regeln zu speziell sind redet man on Overfitting, wenn sie zu allgemein gehalten sind, redet man von underfitting.

                In der Realität kannst du dir das so vorstellen: Ein Modell, das overfittet ist, hat anhand der Daten gelernt, dass jedes Bild, wo ein vierbeiniges Tier auf dem Rasen
                mit einem Tennisball und einer Frisbey zu sehen ist, muss ein Hund sein. Wenn man nun aber ein Bild eines Huskys zeigt, wäre
                das Modell verwirrt, weil es seine Regeln gar nicht richtig anwenden kann.

                Umgekehrt können Regeln aber auch viel zu allgemein und damit das Modell underfittet sein. Die Regel könnte hier lauten, dass alles ein Hund ist, was 4 Beine hat.
                Diese Regel ist offensichtlich nicht sonderlich gut, denn auch Katzen haben meistens 4 Beine.
              </Typography>
            </motion.div>
          </Grid>

          <Grid item>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 1.2 }}
            >
              <Typography align='center' color='primary' variant='h5' p={3}>
                The fancy shit
              </Typography>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1.2 }}
            >
              <iframe
                width="840px"
                height="700px" // Adjust this height as needed for your design
                src="http://127.0.0.1:5000/api/get_plot_html/assets/testplot.pkl"
                title="Embedded Content"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                style={{ border: 'none' }} // Optional: removes the iframe border
              />
            </motion.div>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
};

export default App;
