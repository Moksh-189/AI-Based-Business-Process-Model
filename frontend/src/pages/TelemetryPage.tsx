import TelemetryPanel from '../components/TelemetryPanel';

const TelemetryPage = () => {
    return (
        <div className="h-full flex flex-col gap-6">
            <div>
                <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-emerald-400">
                    Telemetry Dashboard
                </h1>
                <p className="text-gray-400 text-sm mt-1">
                    Real-time KPIs and AI optimization controls
                </p>
            </div>

            <div className="flex-1 min-h-0">
                <TelemetryPanel />
            </div>
        </div>
    );
};

export default TelemetryPage;
