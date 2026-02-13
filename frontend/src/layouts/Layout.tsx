import { Outlet } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import FloatingChatbot from '../components/FloatingChatbot';

const Layout = () => {
    return (
        <div className="flex h-screen w-screen overflow-hidden bg-background text-white selection:bg-primary/30">
            {/* Background Decor */}
            <div className="fixed inset-0 pointer-events-none">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/5 blur-[120px]" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-secondary/5 blur-[120px]" />
            </div>

            {/* Sidebar Navigation */}
            <Sidebar />

            {/* Main Content Area */}
            <main className="flex-1 relative overflow-auto">
                <div className="min-h-full p-8">
                    <Outlet />
                </div>
            </main>

            {/* Global Widgets */}
            <FloatingChatbot />
        </div>
    );
};

export default Layout;
