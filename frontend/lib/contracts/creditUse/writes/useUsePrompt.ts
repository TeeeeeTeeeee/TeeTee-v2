import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';
import { toBigIntSafe } from '../utils';

export const useUsePrompt = () => {
  const { writeContract, data: txHash, isPending: isWriting, error: writeError, reset: resetWrite } = useWriteContract();
  const { data: receipt, isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash: txHash });

  const usePrompt = async (llmId: number | string | bigint) => {
    await writeContract({
      abi: ABI as any,
      address: CONTRACT_ADDRESS as `0x${string}`,
      functionName: 'usePrompt',
      args: [toBigIntSafe(llmId)],
    });
  };

  return { usePrompt, txHash, isWriting, writeError, resetWrite, receipt, isConfirming, isConfirmed } as const;
};

export default useUsePrompt;
