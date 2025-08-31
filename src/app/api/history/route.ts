import { NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';

export async function GET(request: Request) {
  try {
    const { userId } = auth();

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // The URL for the Python API service's history endpoint
    const apiServiceUrl = `http://api:8001/history/${userId}`;

    const apiResponse = await fetch(apiServiceUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!apiResponse.ok) {
      const errorData = await apiResponse.json();
      return NextResponse.json({ error: errorData.detail || 'API request failed' }, { status: apiResponse.status });
    }

    const data = await apiResponse.json();
    return NextResponse.json(data, { status: 200 });

  } catch (error) {
    console.error('Error in /api/history:', error);
    return NextResponse.json({ error: 'An internal server error occurred' }, { status: 500 });
  }
}
