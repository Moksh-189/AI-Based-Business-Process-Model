import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './layouts/Layout';
import Home from './pages/Home';
import Topology from './pages/Topology';
import { ToastProvider } from './context/ToastContext';
import ToastContainer from './components/ToastContainer';

// Placeholder Components for Future Phases
const Workforce = () => <div className="text-2xl font-bold text-gray-500 p-10">Workforce Allocation (Shared View)</div>;
const Telemetry = () => <div className="text-2xl font-bold text-gray-500 p-10">Telemetry Dashboard (Expanded View)</div>;



function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="topology" element={<Topology />} />
            <Route path="workforce" element={<Workforce />} />
            <Route path="telemetry" element={<Telemetry />} />
          </Route>
        </Routes>
        <ToastContainer />
      </BrowserRouter>
    </ToastProvider>
  );
}

export default App;
