import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';

export default function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const API_URL = 'http://localhost:8000';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async () => {
    if (!query.trim() || loading) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    const currentQuery = query;
    setQuery('');
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery })
      });

      const data = await response.json();

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        classification: data.classification
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'error',
        content: 'Failed to connect. Make sure FastAPI is running on port 8000.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-900 flex flex-col">
      <div className="flex-1 flex flex-col w-full">
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full px-4">
              <div className="text-center text-gray-400">
                <h2 className="text-3xl font-semibold mb-4 text-white">Sales Forecasting AI</h2>
                <p className="text-lg">Ask me questions</p>
              </div>
            </div>
          ) : (
            <div className="w-full max-w-4xl mx-auto px-4 py-8 space-y-6">
              {messages.map((msg, i) => (
                <div key={i} className="flex items-start gap-4">
                  <div className={`w-8 h-8 rounded-sm flex items-center justify-center flex-shrink-0 ${
                    msg.role === 'user' ? 'bg-blue-600' : msg.role === 'error' ? 'bg-red-600' : 'bg-green-600'
                  }`}>
                    <span className="text-white text-sm font-semibold">
                      {msg.role === 'user' ? 'U' : msg.role === 'error' ? '!' : 'AI'}
                    </span>
                  </div>
                  <div className="flex-1 pt-1 min-w-0">
                    {msg.classification && (
                      <div className="text-xs font-semibold mb-2 text-gray-500">
                        {msg.classification}
                      </div>
                    )}
                    <div className={`whitespace-pre-wrap break-words ${
                      msg.role === 'error' ? 'text-red-400' : 'text-gray-100'
                    }`}>
                      {msg.content}
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-sm flex items-center justify-center flex-shrink-0 bg-green-600">
                    <span className="text-white text-sm font-semibold">AI</span>
                  </div>
                  <div className="flex-1 pt-1">
                    <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="border-t border-gray-700 bg-gray-800 p-4">
          <div className="max-w-4xl mx-auto flex gap-3 items-center">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Message Sales Forecasting AI..."
              className="flex-1 px-4 py-3 bg-gray-700 text-white border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400"
              disabled={loading}
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              className="px-4 py-3 bg-gray-700 text-gray-400 rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
} 

