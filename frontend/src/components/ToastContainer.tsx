
import { useToast } from '../context/ToastContext';
import { X, CheckCircle, AlertCircle, Info, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';

const ToastContainer = () => {
    const { toasts, removeToast } = useToast();

    return (
        <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3 pointer-events-none">
            {toasts.map((toast) => (
                <ToastItem key={toast.id} {...toast} onRemove={() => removeToast(toast.id)} />
            ))}
        </div>
    );
};

const ToastItem = ({ message, type, onRemove }: { id: string, message: string, type: 'info' | 'success' | 'error' | 'loading', onRemove: () => void }) => {
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        // Trigger enter animation
        requestAnimationFrame(() => setIsVisible(true));
    }, []);

    const handleRemove = () => {
        setIsVisible(false);
        setTimeout(onRemove, 300); // Wait for exit animation
    };

    const icons = {
        info: <Info size={18} className="text-blue-400" />,
        success: <CheckCircle size={18} className="text-emerald-400" />,
        error: <AlertCircle size={18} className="text-red-400" />,
        loading: <Loader2 size={18} className="text-amber-400 animate-spin" />
    };

    const styles = {
        info: 'border-blue-500/20 bg-blue-500/10',
        success: 'border-emerald-500/20 bg-emerald-500/10',
        error: 'border-red-500/20 bg-red-500/10',
        loading: 'border-amber-500/20 bg-amber-500/10'
    };

    return (
        <div
            className={`
                flex items-center gap-3 px-4 py-3 rounded-xl border backdrop-blur-md shadow-lg min-w-[300px] pointer-events-auto
                transition-all duration-300 transform
                ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
                ${styles[type]}
            `}
        >
            {icons[type]}
            <span className="text-sm font-medium text-white/90 flex-1">{message}</span>
            <button onClick={handleRemove} className="text-white/40 hover:text-white transition-colors">
                <X size={16} />
            </button>
        </div>
    );
};

export default ToastContainer;
