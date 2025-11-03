import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import { CONTRACT_ADDRESS } from '@/utils/address';
import ABI from '@/utils/abi.json';

export function useWithdraw() {
  const { data: hash, writeContract, isPending, error, reset } = useWriteContract();
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash });

  const withdraw = async (llmId: bigint) => {
    return writeContract({
      address: CONTRACT_ADDRESS as `0x${string}`,
      abi: ABI,
      functionName: 'withdraw',
      args: [llmId],
    });
  };

  return {
    withdraw,
    isWriting: isPending,
    isConfirming,
    isConfirmed,
    hash,
    error,
    resetWrite: reset,
  };
}

