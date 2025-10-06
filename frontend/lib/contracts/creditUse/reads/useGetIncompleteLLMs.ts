import { useReadContract } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';

export const useGetIncompleteLLMs = () => {
  const { data, isLoading, isError, refetch } = useReadContract({
    abi: ABI as any,
    address: CONTRACT_ADDRESS as `0x${string}`,
    functionName: 'getIncompleteLLMs',
  });

  return { incompleteLLMs: data as bigint[] | undefined, isLoading, isError, refetch };
};

export default useGetIncompleteLLMs;

