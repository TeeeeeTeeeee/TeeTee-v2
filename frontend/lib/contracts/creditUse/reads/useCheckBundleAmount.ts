import { useReadContract } from 'wagmi';
import ABI from '@/utils/abi.json';
import { CONTRACT_ADDRESS } from '@/utils/address';

export const useCheckBundleAmount = () =>
  useReadContract({
    abi: ABI as any,
    address: CONTRACT_ADDRESS as `0x${string}`,
    functionName: 'BUNDLE_AMOUNT',
  });

export default useCheckBundleAmount;
