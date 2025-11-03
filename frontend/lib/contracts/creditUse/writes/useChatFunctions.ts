import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import { CONTRACT_ADDRESS } from '@/utils/address';
import ABI from '@/utils/abi.json';

export function useCreateChatSession() {
  const { data: hash, writeContract, isPending, error, reset } = useWriteContract();
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash });

  const createSession = async () => {
    return writeContract({
      address: CONTRACT_ADDRESS as `0x${string}`,
      abi: ABI,
      functionName: 'createChatSession',
      args: [],
    });
  };

  return {
    createSession,
    isWriting: isPending,
    isConfirming,
    isConfirmed,
    hash,
    error,
    resetWrite: reset,
  };
}

export function useStoreMessageExchange() {
  const { data: hash, writeContract, isPending, error, reset } = useWriteContract();
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash });

  const storeMessages = async (
    sessionId: bigint,
    encryptedUserMessage: string,
    encryptedAIResponse: string
  ) => {
    return writeContract({
      address: CONTRACT_ADDRESS as `0x${string}`,
      abi: ABI,
      functionName: 'storeMessageExchange',
      args: [sessionId, encryptedUserMessage, encryptedAIResponse],
    });
  };

  return {
    storeMessages,
    isWriting: isPending,
    isConfirming,
    isConfirmed,
    hash,
    error,
    resetWrite: reset,
  };
}

export function useDeleteChatSession() {
  const { data: hash, writeContract, isPending, error, reset } = useWriteContract();
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash });

  const deleteSession = async (sessionId: bigint) => {
    return writeContract({
      address: CONTRACT_ADDRESS as `0x${string}`,
      abi: ABI,
      functionName: 'deleteChatSession',
      args: [sessionId],
    });
  };

  return {
    deleteSession,
    isWriting: isPending,
    isConfirming,
    isConfirmed,
    hash,
    error,
    resetWrite: reset,
  };
}

