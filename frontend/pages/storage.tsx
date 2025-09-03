import { useState } from 'react';
import type { FormEvent, ChangeEvent } from 'react';
import Header from '@/components/Header';

// If you have a Header component, you can import and use it here.
// import Header from '../components/Header';

// Matches the success response from /api/upload.ts
type UploadSuccess = {
  filename: string;
  size: number;
  rootHash: string;
  txHash: string;
};

export default function StoragePage() {
  const [rawUploadResponse, setRawUploadResponse] = useState<any>(null);
  // Upload (multipart) state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadLoading, setUploadLoading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string>('');
  const [uploadResult, setUploadResult] = useState<UploadSuccess | null>(null);

  // Download state (stream to browser)
  const [rootHash, setRootHash] = useState<string>('');
  const [downloadFilename, setDownloadFilename] = useState<string>('downloaded-model.bin');
  const [downloadLoading, setDownloadLoading] = useState<boolean>(false);
  const [downloadError, setDownloadError] = useState<string>('');
  const [downloadSuccess, setDownloadSuccess] = useState<boolean>(false);

  // Simulated Chatbot state
  type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string; timestamp: number };
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [chatSaving, setChatSaving] = useState<boolean>(false);
  const [chatSaveError, setChatSaveError] = useState<string>('');
  const [chatSaveResult, setChatSaveResult] = useState<{ filename: string; rootHash: string | undefined; txHash: string } | null>(null);
  const [chatFilename, setChatFilename] = useState<string>('chat_session.txt');

  function handleClearChat() {
    setChatMessages([]);
    setChatSaveResult(null);
    setChatSaveError('');
  }

  function handleChatSend(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const content = chatInput.trim();
    if (!content) return;
    const now = Date.now();
    const userMsg: ChatMessage = { role: 'user', content, timestamp: now };
    const assistantMsg: ChatMessage = {
      role: 'assistant',
      content: `Simulated response: ${content}`,
      timestamp: now + 1,
    };
    setChatMessages((prev) => [...prev, userMsg, assistantMsg]);
    setChatInput('');
  }

  async function handleChatSave() {
    setChatSaving(true);
    setChatSaveError('');
    setChatSaveResult(null);
    try {
      const res = await fetch('/api/chat-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: chatMessages, filename: chatFilename }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error((data as any)?.error || 'Failed to save chat session');
      const result = data as { filename: string; rootHash?: string; txHash: string };
      setChatSaveResult({ filename: result.filename, rootHash: result.rootHash, txHash: result.txHash });
      if (result.rootHash) setRootHash(result.rootHash);
      if (result.filename) setDownloadFilename(result.filename);
    } catch (err: any) {
      setChatSaveError(err?.message || String(err));
    } finally {
      setChatSaving(false);
    }
  }

  async function handleUpload(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setUploadLoading(true);
    setUploadError('');
    setUploadResult(null);

    if (!selectedFile) {
      setUploadLoading(false);
      setUploadError('Please choose a file to upload.');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = (await res.json()) as unknown;
      setRawUploadResponse(data);
      if (!res.ok) throw new Error((data as any)?.error || 'Upload failed');

      const anyData = data as any;
      const tx = anyData?.tx || anyData?.transaction || anyData?.result || anyData?.data || anyData;
      const extractedRoot = (() => {
        try {
          // Direct fields
          if (typeof anyData.rootHash === 'string' && anyData.rootHash) return anyData.rootHash;

          if (typeof tx?.rootHash === 'string' && tx.rootHash) return tx.rootHash;
        } catch {}
        return '';
      })();

      const normalized: UploadSuccess = {
        filename: String(anyData?.filename ?? selectedFile.name ?? 'uploaded-file'),
        size: Number(anyData?.size ?? selectedFile.size ?? 0),
        rootHash: extractedRoot || String(anyData?.rootHash || ''),
        txHash: String(anyData?.txHash || anyData?.tx?.hash || ''),
      };

      setUploadResult(normalized);
      if (normalized.rootHash) setRootHash(normalized.rootHash);
    } catch (err: any) {
      setUploadError(err?.message || String(err));
    } finally {
      setUploadLoading(false);
    }
  }

  async function handleDownload(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setDownloadLoading(true);
    setDownloadError('');
    setDownloadSuccess(false);

    try {
      const url = `/api/download?rootHash=${encodeURIComponent(rootHash)}&filename=${encodeURIComponent(downloadFilename)}`;
      // navigate to the stream URL to trigger browser download
      window.location.href = url;
      setDownloadSuccess(true);
    } catch (err: any) {
      setDownloadError(err?.message || String(err));
    } finally {
      setDownloadLoading(false);
    }
  }

  const safe = (v: unknown): string =>
    typeof v === 'string' || typeof v === 'number'
      ? String(v)
      : (() => {
          try {
            return JSON.stringify(v);
          } catch {
            return String(v);
          }
        })();

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(e.target.files?.[0] ?? null);
  };

  return (
    <div className="min-h-screen grid grid-rows-[auto_1fr]">
      <Header />
      <main className="w-full max-w-3xl mx-auto p-6 sm:p-10">
        <h1 className="text-2xl font-semibold mb-6">0G Storage (Upload / Download)</h1>
        <p className="text-sm text-gray-600 mb-8">
          Upload uses multipart/form-data so you can select a file from your computer. Download streams the file back to your browser.
        </p>

        <section className="mb-12 p-5 border rounded-lg">
          <h2 className="text-xl font-medium mb-4">Upload a File (Select from your computer)</h2>
          <form onSubmit={handleUpload} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Choose File</label>
              <input type="file" className="w-full" onChange={onFileChange} required />
            </div>
            <button
              type="submit"
              disabled={uploadLoading}
              className="px-4 py-2 rounded-md bg-black text-white disabled:opacity-60"
            >
              {uploadLoading ? 'Uploading...' : 'Upload'}
            </button>
          </form>

          {uploadError && <p className="mt-3 text-sm text-red-600">Error: {uploadError}</p>}

          {uploadResult && (
            <div className="mt-4 text-sm break-all">
              <div>
                <span className="font-medium">Filename:</span> {uploadResult.filename} ({uploadResult.size || 0} bytes)
              </div>
              <div>
                <span className="font-medium">Root Hash:</span> {safe(uploadResult.rootHash)}
              </div>
              <div>
                <span className="font-medium">Tx Hash:</span> {safe(uploadResult.txHash)}
              </div>
            </div>
          )}
        </section>

        <section className="mb-12 p-5 border rounded-lg">
          <h2 className="text-xl font-medium mb-4">Simulated Chatbot</h2>
          <div className="space-y-4">
            <div className="border rounded-md p-3 max-h-64 overflow-auto bg-white">
              {chatMessages.length === 0 ? (
                <p className="text-sm text-gray-500">No messages yet. Send a message to start.</p>
              ) : (
                <ul className="space-y-2 text-sm">
                  {chatMessages.map((m, idx) => (
                    <li key={idx} className="flex gap-2">
                      <span className="font-medium">{m.role}:</span>
                      <span className="whitespace-pre-wrap break-words">{m.content}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <form onSubmit={handleChatSend} className="flex gap-2">
              <input
                type="text"
                className="flex-1 px-3 py-2 border rounded-md"
                placeholder="Type a message..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
              />
              <button type="submit" className="px-4 py-2 rounded-md bg-black text-white">Send</button>
            </form>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium mb-1">Save Filename</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border rounded-md"
                  value={chatFilename}
                  onChange={(e) => setChatFilename(e.target.value)}
                />
              </div>
              <div className="flex items-end gap-2">
                <button
                  type="button"
                  onClick={handleChatSave}
                  disabled={chatSaving || chatMessages.length === 0}
                  className="px-4 py-2 rounded-md bg-black text-white disabled:opacity-60"
                >
                  {chatSaving ? 'Saving...' : 'End & Save Session'}
                </button>
                <button
                  type="button"
                  onClick={handleClearChat}
                  className="px-4 py-2 rounded-md border"
                >
                  Clear
                </button>
              </div>
            </div>

            {chatSaveError && <p className="text-sm text-red-600">Error: {chatSaveError}</p>}

            {chatSaveResult && (
              <div className="text-sm break-all">
                <div><span className="font-medium">Saved Filename:</span> {chatSaveResult.filename}</div>
                <div><span className="font-medium">Root Hash:</span> {safe(chatSaveResult.rootHash)}</div>
                <div><span className="font-medium">Tx Hash:</span> {safe(chatSaveResult.txHash)}</div>
                <div className="mt-2 text-gray-600">You can use the Root Hash in the Download section below to retrieve the transcript.</div>
              </div>
            )}
          </div>
        </section>

        <section className="p-5 border rounded-lg">
          <h2 className="text-xl font-medium mb-4">Download a File (Browser Download)</h2>
          <form onSubmit={handleDownload} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Root Hash</label>
              <input
                type="text"
                className="w-full px-3 py-2 border rounded-md"
                placeholder="0x..."
                value={rootHash}
                onChange={(e) => setRootHash(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Download Filename</label>
              <input
                type="text"
                className="w-full px-3 py-2 border rounded-md"
                placeholder="downloaded-model.bin"
                value={downloadFilename}
                onChange={(e) => setDownloadFilename(e.target.value)}
                required
              />
            </div>
            <button
              type="submit"
              disabled={downloadLoading}
              className="px-4 py-2 rounded-md bg-black text-white disabled:opacity-60"
            >
              {downloadLoading ? 'Preparing...' : 'Download'}
            </button>
          </form>

          {downloadError && <p className="mt-3 text-sm text-red-600">Error: {downloadError}</p>}

          {downloadSuccess && (
            <p className="mt-3 text-sm text-green-700">
              If the download didn't start automatically, check your browser's popup/download settings.
            </p>
          )}
        </section>
      </main>
    </div>
  );
}