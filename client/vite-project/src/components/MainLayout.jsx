import { Grid, Box, Typography, Card, CardContent, Button, LinearProgress } from "@mui/material";
import { CloudUpload as UploadIcon } from "@mui/icons-material";
import Header from "./Header.jsx";
import UploadForm from "./UploadForm.jsx";

const MainLayout = ({ onFileChange, onSubmit, loading, error }) => {
    return (
      <Grid
        container
        spacing={4}
        alignItems="center"
        justifyContent="center"
        sx={{ height: "100vh", padding: "20px" }}
      >
        <Grid item xs={12} md={6}>
          <Header />
        </Grid>
        <Grid item xs={12} md={6}>
          <UploadForm
            onFileChange={onFileChange}
            onSubmit={onSubmit}
            loading={loading}
            error={error}
          />
        </Grid>
      </Grid>
    );
  };

  export default MainLayout