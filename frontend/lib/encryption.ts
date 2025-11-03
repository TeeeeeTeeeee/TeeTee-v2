/**
 * Simple encryption utilities for chat messages
 * Messages are encrypted with the user's address as the key
 * Only the wallet owner can decrypt their messages
 */

import CryptoJS from 'crypto-js';

/**
 * Encrypt a message using the user's wallet address
 */
export function encryptMessage(message: string, walletAddress: string): string {
  if (!message || !walletAddress) {
    throw new Error('Message and wallet address are required');
  }
  
  // Use wallet address as encryption key (deterministic)
  const encrypted = CryptoJS.AES.encrypt(message, walletAddress.toLowerCase()).toString();
  return encrypted;
}

/**
 * Decrypt a message using the user's wallet address
 */
export function decryptMessage(encryptedMessage: string, walletAddress: string): string {
  if (!encryptedMessage || !walletAddress) {
    throw new Error('Encrypted message and wallet address are required');
  }
  
  try {
    const decrypted = CryptoJS.AES.decrypt(encryptedMessage, walletAddress.toLowerCase());
    const message = decrypted.toString(CryptoJS.enc.Utf8);
    
    if (!message) {
      throw new Error('Decryption failed - invalid key or corrupted data');
    }
    
    return message;
  } catch (error) {
    console.error('Decryption error:', error);
    throw new Error('Failed to decrypt message');
  }
}

