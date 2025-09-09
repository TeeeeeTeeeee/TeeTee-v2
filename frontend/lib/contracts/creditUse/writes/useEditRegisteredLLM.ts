import { useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';
import { toBigIntSafe } from '../utils';

// Note: function name in contract appears to be 'editRegistedLLM' (typo). We mirror that here.
export const useEditRegisteredLLM = () => {
  const { writeContract, data: txHash, isPending: isWriting, error: writeError, reset: resetWrite } = useWriteContract();
  const { data: receipt, isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash: txHash });

  const editRegisteredLLM = async (
    id: number | string | bigint,
    host1: string,
    host2: string,
    shardUrl1: string,
    shardUrl2: string,
    modelName: string,
  ) => {
    await writeContract({
      abi: ABI as any,
      address: CONTRACT_ADDRESS as `0x${string}`,
      functionName: 'editRegistedLLM',
      args: [toBigIntSafe(id), host1, host2, shardUrl1, shardUrl2, modelName],
    });
  };

  return { editRegisteredLLM, txHash, isWriting, writeError, resetWrite, receipt, isConfirming, isConfirmed } as const;
};

export default useEditRegisteredLLM;
