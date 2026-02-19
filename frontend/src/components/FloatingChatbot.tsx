import React, { useState, useRef, useEffect, useCallback } from 'react';
import { WS_URL } from '../config/api';
import { MessageSquare, X, Send, Bot } from 'lucide-react';

interface Message {
    id: number;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
}

const FloatingChatbot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const INTRO_MESSAGE: Message = {
        id: 1,
        text: "Hello! I'm your AI process assistant. How can I help you optimize your workflow today?",
        sender: 'bot',
        timestamp: new Date()
    };

    const [messages, setMessages] = useState<Message[]>([INTRO_MESSAGE]);
    const [inputValue, setInputValue] = useState("");
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    const [isLoading, setIsLoading] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
    const ws = useRef<WebSocket | null>(null);

    const connectWebSocket = useCallback(() => {
        if (ws.current?.readyState === WebSocket.OPEN) return;

        setConnectionStatus('connecting');
        const socket = new WebSocket(`${WS_URL}/ws/chat`);
        ws.current = socket;

        socket.onopen = () => {
            console.log("Connected to Chatbot WS");
            setConnectionStatus('connected');
        };

        socket.onmessage = (event) => {
            const message = event.data;
            setMessages(prev => [...prev, {
                id: Date.now(),
                text: message,
                sender: 'bot',
                timestamp: new Date()
            }]);
            setIsLoading(false);
        };

        socket.onclose = () => {
            console.log("Chatbot WS Disconnected");
            setConnectionStatus('disconnected');
        };

        socket.onerror = () => {
            setConnectionStatus('disconnected');
        };
    }, []);

    useEffect(() => {
        connectWebSocket();
        return () => {
            ws.current?.close();
        };
    }, [connectWebSocket]);

    const handleSendMessage = (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!inputValue.trim() || isLoading) return;

        const userMsg: Message = {
            id: Date.now(),
            text: inputValue,
            sender: 'user',
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMsg]);
        setInputValue("");
        setIsLoading(true);

        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(userMsg.text);
        } else {
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                text: "⚠️ Disconnected. Reconnecting...",
                sender: 'bot',
                timestamp: new Date()
            }]);
            setIsLoading(false);
            connectWebSocket(); // Try to reconnect
        }
    };

    // Auto-focus input on mount
    useEffect(() => {
        const timer = setTimeout(() => {
            // Ensure connection is established
            if (ws.current?.readyState !== WebSocket.OPEN) {
                connectWebSocket();
            }
        }, 1000);
        return () => clearTimeout(timer);
    }, [connectWebSocket]);

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
            {/* Chat Window */}
            <div
                className={`
          transition-all duration-300 ease-in-out origin-bottom-right
          ${isOpen
                        ? 'w-96 h-[500px] opacity-100 scale-100'
                        : 'w-0 h-0 opacity-0 scale-50 pointer-events-none'
                    }
          bg-surface/90 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col mb-4
        `}
            >
                {/* Header */}
                <div className="p-4 border-b border-white/10 bg-primary/10 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                            <Bot size={18} />
                        </div>
                        <div>
                            <h3 className="font-bold text-sm">AI Assistant</h3>
                            <div className="flex items-center gap-1.5">
                                <span className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-emerald-500 animate-pulse' :
                                    connectionStatus === 'connecting' ? 'bg-amber-500 animate-pulse' :
                                        'bg-red-500'
                                    }`}></span>
                                <span className="text-xs text-gray-400">
                                    {connectionStatus === 'connected' ? 'Online' :
                                        connectionStatus === 'connecting' ? 'Connecting...' :
                                            'Offline'}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setMessages([INTRO_MESSAGE])}
                            className="p-1 hover:bg-white/10 rounded-lg transition-colors text-xs text-gray-400"
                            title="Clear Chat"
                        >
                            Clear
                        </button>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="p-1 hover:bg-white/10 rounded-lg transition-colors"
                        >
                            <X size={18} className="text-gray-400" />
                        </button>
                    </div>
                </div>

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
                    {connectionStatus === 'disconnected' && (
                        <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-300">
                            ⚠️ Disconnected from server.
                            <button
                                onClick={connectWebSocket}
                                className="ml-2 text-primary underline hover:text-primary/80"
                            >
                                Reconnect
                            </button>
                        </div>
                    )}
                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                data-testid="chat-message"
                                className={`
                  max-w-[80%] p-3 rounded-2xl text-sm leading-relaxed
                  ${msg.sender === 'user'
                                        ? 'bg-primary text-background rounded-tr-none'
                                        : 'bg-white/5 border border-white/10 rounded-tl-none'
                                    }
                `}
                            >
                                {msg.text}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-white/5 border border-white/10 rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-1.5">
                                <span className="w-2 h-2 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <span className="w-2 h-2 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <span className="w-2 h-2 bg-primary/70 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <form onSubmit={handleSendMessage} className="p-4 border-t border-white/10 bg-surface/50">
                    <div className="flex items-center gap-2">
                        <input
                            type="text"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            placeholder="Ask about process bottlenecks... (or enter API Key)"
                            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-primary/50 transition-colors placeholder:text-gray-500"
                        />
                        <button
                            type="submit"
                            disabled={!inputValue.trim()}
                            className="p-2.5 bg-primary text-background rounded-xl hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            <Send size={18} />
                        </button>
                    </div>
                    <div className="text-[10px] text-gray-500 mt-2 text-center">
                        If asked, paste your Google Gemini API Key above.
                    </div>
                </form>
            </div>

            {/* Toggle Button */}
            <button
                data-testid="chatbot-toggle"
                onClick={() => setIsOpen(!isOpen)}
                className={`
          p-4 rounded-full shadow-lg transition-all duration-300 hover:scale-110 active:scale-95
          ${isOpen
                        ? 'bg-surface border border-white/10 text-gray-400 rotate-90'
                        : 'bg-primary text-background shadow-[0_0_20px_rgba(6,182,212,0.4)]'
                    }
        `}
            >
                {isOpen ? <X size={24} /> : <MessageSquare size={24} />}
            </button>
        </div>
    );
};

export default FloatingChatbot;
