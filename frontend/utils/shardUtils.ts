/**
 * Utility functions for shard management and host slot determination
 */

export type ShardType = 'shard1' | 'shard2';

export interface HostSlotInfo {
  host1Empty: boolean;
  host2Empty: boolean;
  availableShard: ShardType | null;
  isComplete: boolean;
}

/**
 * Check if an address is empty (null or zero address)
 */
export const isAddressEmpty = (address: string | undefined): boolean => {
  return !address || address === '0x0000000000000000000000000000000000000000';
};

/**
 * Determine which host slots are available for a model
 */
export const getHostSlotInfo = (
  host1: string | undefined,
  host2: string | undefined
): HostSlotInfo => {
  const host1Empty = isAddressEmpty(host1);
  const host2Empty = isAddressEmpty(host2);
  
  // Determine which shard is available
  let availableShard: ShardType | null = null;
  if (host1Empty && !host2Empty) {
    availableShard = 'shard1';
  } else if (!host1Empty && host2Empty) {
    availableShard = 'shard2';
  } else if (host1Empty && host2Empty) {
    // Both empty - can choose either (but this shouldn't happen in join scenario)
    availableShard = null;
  }
  
  const isComplete = !host1Empty && !host2Empty;
  
  return {
    host1Empty,
    host2Empty,
    availableShard,
    isComplete,
  };
};

/**
 * Get the shard name for display
 */
export const getShardDisplayName = (shard: ShardType, includeLayers: boolean = true): string => {
  if (shard === 'shard1') {
    return includeLayers ? 'Shard 1 (Lower Layers)' : 'Shard 1';
  }
  return includeLayers ? 'Shard 2 (Upper Layers)' : 'Shard 2';
};

/**
 * Get the shard layer range for display
 */
export const getShardLayerRange = (shard: ShardType): string => {
  return shard === 'shard1' ? '1-50' : '51-100';
};

/**
 * Determine if user is hosting a specific model
 */
export const isUserHostingModel = (
  userAddress: string,
  host1: string | undefined,
  host2: string | undefined
): boolean => {
  if (!userAddress) return false;
  const userLower = userAddress.toLowerCase();
  const host1Lower = (host1 || '').toLowerCase();
  const host2Lower = (host2 || '').toLowerCase();
  
  return host1Lower === userLower || host2Lower === userLower;
};

/**
 * Get the user's role in hosting (host1 or host2)
 */
export const getUserHostRole = (
  userAddress: string,
  host1: string | undefined,
  host2: string | undefined
): 'host1' | 'host2' | null => {
  if (!userAddress) return null;
  const userLower = userAddress.toLowerCase();
  const host1Lower = (host1 || '').toLowerCase();
  const host2Lower = (host2 || '').toLowerCase();
  
  if (host1Lower === userLower) return 'host1';
  if (host2Lower === userLower) return 'host2';
  return null;
};

/**
 * Validate shard selection when joining a model
 */
export const validateShardSelection = (
  selectedShard: ShardType,
  availableShard: ShardType | null
): { valid: boolean; error?: string } => {
  if (!availableShard) {
    return { valid: false, error: 'No shard slots available' };
  }
  
  if (selectedShard !== availableShard) {
    const availableShardName = getShardDisplayName(availableShard, false);
    return {
      valid: false,
      error: `${availableShardName} is already taken. Please select the available shard.`,
    };
  }
  
  return { valid: true };
};

/**
 * Normalize URL for comparison (removes trailing slashes, converts to lowercase)
 */
export const normalizeUrl = (url: string): string => {
  return url.trim().replace(/\/+$/, '').toLowerCase();
};

/**
 * Check if two URLs are the same (after normalization)
 */
export const areUrlsSame = (url1: string, url2: string): boolean => {
  return normalizeUrl(url1) === normalizeUrl(url2);
};

/**
 * Validate that a URL is different from an existing URL
 */
export const validateUniqueUrl = (
  newUrl: string,
  existingUrl: string
): { valid: boolean; error?: string } => {
  if (!newUrl || !existingUrl) {
    return { valid: true };
  }
  
  if (areUrlsSame(newUrl, existingUrl)) {
    return {
      valid: false,
      error: 'This TEE endpoint URL is already in use by the first host. Please use a different endpoint.',
    };
  }
  
  return { valid: true };
};

