import React from "react";
import {
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Box,
} from "@mui/material";
import { CloudUpload as UploadIcon } from "@mui/icons-material";

const UploadForm = ({ onFileChange, onSubmit, loading, error }) => {
  return (
    <Card
      className="upload-form"
      sx={{
        position: "absolute",
        right: "10%",
        top: "50%",
        transform: "translateY(-50%)",
        animation: "slideInRight 1s ease-out",
      }}
    >
      <CardContent>
        <Typography variant="h5" mb={2} fontWeight="bold" sx={{ color: "#81C784" }}>
          Upload Your Video for Analysis
        </Typography>
        <Box textAlign="center" mb={2}>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => onFileChange(e.target.files[0])}
            style={{
              margin: "10px 0",
              padding: "10px",
              borderRadius: "5px",
              border: "1px solid #ccc",
              width: "100%",
              color: "#fff",
              backgroundColor: "rgba(0, 0, 0, 0.2)",
            }}
          />
          <Button
            variant="contained"
            color="primary"
            startIcon={<UploadIcon />}
            onClick={onSubmit}
            sx={{
              mt: 2,
              background: "linear-gradient(to right, #4CAF50, #81C784)",
              color: "#fff",
              '&:hover': {
                background: "linear-gradient(to left, #4CAF50, #388E3C)",
              },
            }}
          >
            Analyze Video
          </Button>
        </Box>
        {loading && (
          <Box mt={2}>
            <Typography sx={{ color: "#fff" }}>Processing...</Typography>
            <LinearProgress />
          </Box>
        )}
        {error && <Typography color="error">{error}</Typography>}
      </CardContent>
    </Card>
  );
};

export default UploadForm;
