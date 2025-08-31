import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { url, prompt } = await request.json();

    if (!url || !prompt) {
      return NextResponse.json({ error: 'URL and prompt are required' }, { status: 400 });
    }

    // The URL for the Python API service, as defined in docker-compose.yml
    const apiServiceUrl = 'http://api:8001/scrape';

    const apiResponse = await fetch(apiServiceUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url, prompt }),
    });

    if (!apiResponse.ok) {
      const errorData = await apiResponse.json();
      return NextResponse.json({ error: errorData.detail || 'API request failed' }, { status: apiResponse.status });
    }

    const data = await apiResponse.json();
    return NextResponse.json(data, { status: 200 });

  } catch (error) {
    console.error('Error in /api/scrape:', error);
    return NextResponse.json({ error: 'An internal server error occurred' }, { status: 500 });
  }
}
