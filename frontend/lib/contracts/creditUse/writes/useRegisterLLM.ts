import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';

export const useRegisterLLM = () => {
  const { writeContract, data: txHash, isPending: isWriting, error: writeError, reset: resetWrite } = useWriteContract();
  const { data: receipt, isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash: txHash });

  const registerLLM = async (
    llmId: number,           // Pass array length for new, or existing ID to update
    host1: string, 
    host2: string,
    shardUrl1: string, 
    shardUrl2: string,
    modelName: string
  ) => {
    await writeContract({
      abi: ABI as any,
      address: CONTRACT_ADDRESS as `0x${string}`,
      functionName: 'registerLLM',
      args: [llmId, host1, host2, shardUrl1, shardUrl2, modelName],
    });
  };

  return { registerLLM, txHash, isWriting, writeError, resetWrite, receipt, isConfirming, isConfirmed } as const;
};

export default useRegisterLLM;
