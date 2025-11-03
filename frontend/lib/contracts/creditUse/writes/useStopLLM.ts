import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import { CONTRACT_ADDRESS } from '@/utils/address';
import ABI from '@/utils/abi.json';

export function useStopLLM() {
  const { data: hash, writeContract, isPending, error, reset } = useWriteContract();
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash });

  const stopLLM = async (llmId: number, hostNumber: number) => {
    return writeContract({
      address: CONTRACT_ADDRESS as `0x${string}`,
      abi: ABI,
      functionName: 'stopLLM',
      args: [BigInt(llmId), hostNumber],
    });
  };

  return {
    stopLLM,
    isWriting: isPending,
    isConfirming,
    isConfirmed,
    hash,
    error,
    resetWrite: reset,
  };
}

