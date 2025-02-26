import React from "react";
import {
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Button,
  Box,
} from "@mui/material";
import { FaSmile, FaMicrophone, FaLightbulb } from "react-icons/fa";
import { Replay as ReplayIcon } from "@mui/icons-material";

const AnalysisResults = ({ result, onGoBack }) => {
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh", // Center vertically
        padding: "20px", // Add some padding for responsiveness
      }}
    >
      <Grid container spacing={3} maxWidth="md">
        <Grid item xs={12}>
          <Typography variant="h4" textAlign="center" fontWeight="bold">
            Analysis Results
          </Typography>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" display="flex" alignItems="center">
                <FaSmile style={{ marginRight: "10px", color: "#4caf50" }} /> Face Analysis
              </Typography>
              <Typography variant="body1" mt={2}>
                Confidence Level: {result.face_confidence}
              </Typography>
              <LinearProgress variant="determinate" value={parseFloat(result.face_confidence)} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" display="flex" alignItems="center">
                <FaMicrophone style={{ marginRight: "10px", color: "#2196f3" }} /> Voice Analysis
              </Typography>
              <Typography variant="body1" mt={2}>
                Confidence Level: {result.voice_confidence}
              </Typography>
              <LinearProgress variant="determinate" value={parseFloat(result.voice_confidence)} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" display="flex" alignItems="center">
                <FaLightbulb style={{ marginRight: "10px", color: "#ff9800" }} /> Overall Confidence
              </Typography>
              <Typography variant="body1" mt={2}>
                {result.confidence}
              </Typography>
              <LinearProgress variant="determinate" value={parseFloat(result.confidence)} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold">
                Transcribed Text:
              </Typography>
              <Typography variant="body1" color="textSecondary" mt={1}>
                {result.transcribed_text}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold">
                Suggestions for Improvement:
              </Typography>
              <ul>
                {result.suggestions.map((suggestion, index) => (
                  <li key={index}>
                    <FaLightbulb style={{ marginRight: "10px", color: "#ffc107" }} />
                    {suggestion}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} textAlign="center">
          <Button
            variant="contained"
            color="secondary"
            startIcon={<ReplayIcon />}
            onClick={onGoBack}
          >
            Analyze Another Video
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalysisResults;
