import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';

export const useBuyCredits = () => {
  const { writeContract, data: txHash, isPending: isWriting, error: writeError, reset: resetWrite } = useWriteContract();
  const { data: receipt, isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash: txHash });

  const buyCredits = async (valueWei: bigint) => {
    await writeContract({
      abi: ABI as any,
      address: CONTRACT_ADDRESS as `0x${string}`,
      functionName: 'buyCredits',
      args: [],
      value: valueWei,
    });
  };

  return { buyCredits, txHash, isWriting, writeError, resetWrite, receipt, isConfirming, isConfirmed } as const;
};

export default useBuyCredits;
