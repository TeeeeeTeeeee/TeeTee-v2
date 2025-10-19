import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

// Define the data directory path (in current working directory)
// Next.js runs from the frontend directory, so process.cwd() is already C:\Code\TeeTee-v2\frontend
const DATA_DIR = path.join(process.cwd(), 'data');
const SESSIONS_FILE = path.join(DATA_DIR, 'chat-sessions.json');

export type ChatSession = {
  _id: string;
  walletAddress: string;
  filename: string;
  preview?: string; // First user message preview
  rootHash: string | null;
  txHash: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
};

type SessionsData = {
  sessions: ChatSession[];
};

// Ensure data directory exists
function ensureDataDir(): void {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
}

// Initialize sessions file if it doesn't exist
function ensureSessionsFile(): void {
  ensureDataDir();
  if (!fs.existsSync(SESSIONS_FILE)) {
    const initialData: SessionsData = { sessions: [] };
    fs.writeFileSync(SESSIONS_FILE, JSON.stringify(initialData, null, 2), 'utf-8');
  }
}

// Read all sessions from JSON file
function readSessions(): ChatSession[] {
  ensureSessionsFile();
  try {
    const data = fs.readFileSync(SESSIONS_FILE, 'utf-8');
    const parsed: SessionsData = JSON.parse(data);
    return parsed.sessions || [];
  } catch (error) {
    console.error('Error reading sessions file:', error);
    return [];
  }
}

// Write sessions to JSON file
function writeSessions(sessions: ChatSession[]): void {
  ensureDataDir();
  const data: SessionsData = { sessions };
  fs.writeFileSync(SESSIONS_FILE, JSON.stringify(data, null, 2), 'utf-8');
}

// Get sessions for a specific wallet address
export function getSessionsByWallet(walletAddress: string): ChatSession[] {
  const allSessions = readSessions();
  return allSessions
    .filter((s) => s.walletAddress.toLowerCase() === walletAddress.toLowerCase())
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
}

// Add a new session
export function addSession(session: Omit<ChatSession, '_id' | 'createdAt' | 'updatedAt'>): ChatSession {
  const sessions = readSessions();
  const now = new Date().toISOString();
  const newSession: ChatSession = {
    ...session,
    _id: randomUUID(),
    createdAt: now,
    updatedAt: now,
  };
  sessions.push(newSession);
  writeSessions(sessions);
  return newSession;
}

// Update an existing session
export function updateSession(id: string, updates: Partial<ChatSession>): ChatSession | null {
  const sessions = readSessions();
  const index = sessions.findIndex((s) => s._id === id);
  
  if (index === -1) {
    return null;
  }
  
  sessions[index] = {
    ...sessions[index],
    ...updates,
    updatedAt: new Date().toISOString(),
  };
  
  writeSessions(sessions);
  return sessions[index];
}

// Delete a session
export function deleteSession(id: string): boolean {
  const sessions = readSessions();
  const filteredSessions = sessions.filter((s) => s._id !== id);
  
  if (filteredSessions.length === sessions.length) {
    return false; // Session not found
  }
  
  writeSessions(filteredSessions);
  return true;
}

// Get all sessions (for admin purposes)
export function getAllSessions(): ChatSession[] {
  return readSessions();
}
