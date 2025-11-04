import { MongoClient, Db, ServerApiVersion } from 'mongodb';

const uri = process.env.MONGODB_URI;

// Extend the global type to include our MongoDB client promise
declare global {
  // eslint-disable-next-line no-var
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

let client: MongoClient;
let clientPromise: Promise<MongoClient> | null = null;

// MongoDB connection options with Stable API
const options = {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
  serverSelectionTimeoutMS: 10000, // Timeout after 10s
  socketTimeoutMS: 45000,
};

// Only initialize MongoDB if URI is provided
if (uri) {
  if (process.env.NODE_ENV === 'development') {
    // In development mode, use a global variable so the connection
    // is not repeatedly created during hot reloads
    if (!global._mongoClientPromise) {
      client = new MongoClient(uri, options);
      global._mongoClientPromise = client.connect();
    }
    clientPromise = global._mongoClientPromise;
  } else {
    // In production mode, create a new client
    client = new MongoClient(uri, options);
    clientPromise = client.connect();
  }
}

// Export the database connection function
export async function connectToDatabase(): Promise<{
  client: MongoClient;
  db: Db;
}> {
  if (!clientPromise) {
    throw new Error('MongoDB is not configured. Add MONGODB_URI to .env.local to enable MongoDB storage.');
  }
  
  try {
    const client = await clientPromise;
    const db = client.db('TeeTee');
    return { client, db };
  } catch (error) {
    console.error('Failed to connect to database:', error);
    throw error;
  }
}

// Export the client promise for direct access if needed
export default clientPromise;

