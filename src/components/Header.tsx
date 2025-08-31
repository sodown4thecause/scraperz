"use client";

import { UserButton } from "@clerk/nextjs";
import Link from "next/link";
import { Button } from "./ui/button";

const Header = () => {
  return (
    <header className="bg-white shadow-md">
      <div className="mx-auto max-w-7xl px-4 py-2 sm:px-6 lg:px-8 flex justify-between items-center">
        <div className="flex items-center gap-8">
            <h1 className="text-xl font-bold text-gray-900">Intelligent Scraper</h1>
            <nav className="flex items-center gap-4">
                <Link href="/">
                    <Button variant="ghost">Dashboard</Button>
                </Link>
                <Link href="/monitoring">
                    <Button variant="ghost">Monitoring</Button>
                </Link>
                <Link href="/history">
                    <Button variant="ghost">History</Button>
                </Link>
            </nav>
        </div>
        <UserButton afterSignOutUrl="/" />
      </div>
    </header>
  );
};

export default Header;
