import React from "react";
import { Grid, Box, Typography, Card, CardContent, Button, LinearProgress } from "@mui/material";
import { CloudUpload as UploadIcon } from "@mui/icons-material";

const Header = () => {
  return (
    <Box
      textAlign="center"
      mb={4}
      sx={{
        animation: "fadeInDown 1s ease-out",
        color: "white",
      }}
    >
      <Typography
        variant="h3"
        fontWeight="bold"
        sx={{
          background: "linear-gradient(to right, #4CAF50, #81C784)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          animation: "textGlow 2s infinite alternate",
        }}
      >
        Confidence Assessment Tool for Virtual Interviews
      </Typography>
      <Typography
        variant="body1"
        color="textSecondary"
        mt={2}
        sx={{ color: "#B0BEC5" }}
      >
        Leverage AI to understand and improve your interview confidence with
        detailed feedback and actionable insights.
      </Typography>
    </Box>
  );
};

export default Header;
