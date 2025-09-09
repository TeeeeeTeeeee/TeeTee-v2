import { useReadContract } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';

export const useCheckOwner = () =>
  useReadContract({
    abi: ABI as any,
    address: CONTRACT_ADDRESS as `0x${string}`,
    functionName: 'owner',
  });

export default useCheckOwner;
