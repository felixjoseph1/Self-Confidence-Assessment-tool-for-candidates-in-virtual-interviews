import React, { useState } from "react";
import axios from "axios";
import { Container } from "@mui/material";
import MainLayout from "./components/MainLayout";
import AnalysisResults from "./components/AnalysisResults";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (file) => {
    setSelectedFile(file);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      alert("Please select a video file.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://localhost:5000/analyze", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(response.data);
    } catch (err) {
      setError("An error occurred while analyzing the video.");
    } finally {
      setLoading(false);
    }
  };

  const handleGoBack = () => {
    setResult(null);
    setSelectedFile(null);
  };

  return (
    <Container maxWidth="lg">
      {/* Render MainLayout when no result is available */}
      {!result ? (
        <MainLayout
          onFileChange={handleFileChange}
          onSubmit={handleSubmit}
          loading={loading}
          error={error}
        />
      ) : (
        // Render AnalysisResults when the result is available
        <AnalysisResults result={result} onGoBack={handleGoBack} />
      )}
    </Container>
  );
}

export default App;
