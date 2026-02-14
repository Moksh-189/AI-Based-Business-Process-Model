import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './layouts/Layout';
import Home from './pages/Home';
import Topology from './pages/Topology';
import WorkforcePage from './pages/WorkforcePage';
import TelemetryPage from './pages/TelemetryPage';
import { ToastProvider } from './context/ToastContext';
import ToastContainer from './components/ToastContainer';

function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="topology" element={<Topology />} />
            <Route path="workforce" element={<WorkforcePage />} />
            <Route path="telemetry" element={<TelemetryPage />} />
          </Route>
        </Routes>
        <ToastContainer />
      </BrowserRouter>
    </ToastProvider>
  );
}

export default App;
