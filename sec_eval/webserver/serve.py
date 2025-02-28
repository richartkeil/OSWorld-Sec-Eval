#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
from pathlib import Path

# Configuration
PORT = 8000
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def main():
    try:
        # Create the server
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            # Print helpful information
            print(f"\n📧 Email Mockup Server")
            print(f"{'=' * 50}")
            print(f"🌍 Server started at: http://localhost:{PORT}")
            print(f"📁 Serving from: {DIRECTORY}")
            print(f"📋 Available pages:")
            print(f"   • http://localhost:{PORT}/mailbox.html")
            print(f"\n💡 Press Ctrl+C to stop the server")
            print(f"{'=' * 50}\n")

            # Open the mailbox in the default browser
            webbrowser.open(f"http://localhost:{PORT}/mailbox.html")

            # Start the server
            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Have a great day!")
        
if __name__ == "__main__":
    main() 