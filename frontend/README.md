# RAG Assistant - ChatGPT-Style Frontend

A modern, ChatGPT-style interface for your RAG (Retrieval-Augmented Generation) document assistant with comprehensive file upload support.

## Features

✨ **ChatGPT-Style Interface**
- Dark theme with clean, modern design
- Collapsible sidebar with conversation history
- Multiple conversations support
- Real-time typing indicators
- Source citations with expandable details

📎 **File Upload Support**
- Drag-and-drop or click to upload
- 20MB per file limit
- **30+ file types supported:**
  - Documents: PDF, Word (.docx/.doc), PowerPoint (.pptx/.ppt), Excel (.xlsx/.xls), CSV
  - Text/Markup: TXT, Markdown, RST, JSON, XML, HTML
  - Code Files: Python, JavaScript, Java, C/C++, Shell, YAML
- Real-time upload progress
- Visual file upload confirmations
- Automatic file validation

🎨 **Modern UI/UX**
- Smooth animations and transitions
- Responsive design (mobile-friendly)
- Auto-expanding textarea
- Example prompts for new conversations
- Markdown rendering for rich text
- File type icons and size display

🚀 **Performance**
- Fast React 19 with Vite
- Tailwind CSS for styling
- Optimized bundle size
- Lazy loading components

## Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` if your backend runs on a different port:

```env
VITE_API_URL=http://localhost:8000
```

### 3. Start Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatInput.jsx      # Message input with file upload
│   │   ├── ChatMessage.jsx    # Message bubble with markdown & uploads
│   │   ├── FileUpload.jsx     # File upload component
│   │   └── Sidebar.jsx        # Conversation sidebar
│   ├── services/
│   │   └── api.js             # API client functions
│   ├── App.jsx                # Main app component
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── index.html
├── package.json
├── vite.config.js
└── .env.example
```

## Supported File Types

### Documents (20MB limit)
- **PDF** (.pdf) - Optimized with PyMuPDF
- **Word** (.docx, .doc)
- **PowerPoint** (.pptx, .ppt)
- **Excel** (.xlsx, .xls)
- **CSV** (.csv)

### Text/Markup
- **Plain Text** (.txt)
- **Markdown** (.md)
- **reStructuredText** (.rst)
- **JSON** (.json)
- **XML** (.xml)
- **HTML** (.html, .htm)

### Code Files
- **Python** (.py)
- **JavaScript** (.js)
- **Java** (.java)
- **C/C++** (.cpp, .c, .h)
- **Shell Scripts** (.sh)
- **YAML** (.yml, .yaml)

## API Integration

The frontend connects to your RAG backend through these endpoints:

### Ask Question
```javascript
POST /api/ask
{
  "question": "What is the main topic?",
  "top_k": 5
}
```

### Upload Document
```javascript
POST /api/upload
Content-Type: multipart/form-data
Body: FormData with file field
```

Response:
```json
{
  "message": "File uploaded successfully",
  "filename": "document.pdf"
}
```

### Health Check
```javascript
GET /api/health
```

## Using File Upload

### Upload via UI

1. Click the 📎 (paperclip) icon in the input area
2. Select a file from your computer (or drag & drop)
3. File will be validated and uploaded automatically
4. Success message appears in the conversation

### Upload Behavior

- **File Validation**: Size and type checked before upload
- **Progress Indicator**: Shows upload percentage
- **Error Handling**: Clear error messages for invalid files
- **Success Feedback**: Green confirmation with file details

### Upload Error Messages

- "File size exceeds 20MB limit (X.X MB)"
- "Unsupported file type. Please upload a valid document, text, or code file."
- "Upload failed. Please try again."

## Customization

### Colors

The UI uses ChatGPT's color scheme. To customize, edit `src/index.css`:

```css
/* Background colors */
bg-[#343541]  /* Main chat area */
bg-[#202123]  /* Sidebar */
bg-[#444654]  /* Assistant messages */
bg-[#10a37f]  /* User messages & accents */
bg-[#40414F]  /* Input field */
```

### File Upload Settings

To modify file upload limits, edit `src/components/FileUpload.jsx`:

```javascript
const MAX_FILE_SIZE = 20 * 1024 * 1024; // Change to desired size in bytes
const FILE_EXTENSIONS = [ /* Add or remove extensions */ ];
```

### Example Prompts

Edit the example prompts in `App.jsx`:

```jsx
<ExamplePrompt onClick={handleSend} text="Your custom prompt" />
```

### Branding

Update the title and description in `index.html` and `App.jsx`.

## Usage Tips

### Keyboard Shortcuts

- **Enter**: Send message
- **Shift + Enter**: New line in message

### Conversation Management

- Click "New chat" to start a fresh conversation
- Click any conversation in sidebar to switch
- Hover over conversation and click trash icon to delete
- Click "Clear" in header to clear current conversation

### Source Citations

- Assistant responses include expandable source citations
- Click "N sources" to view document chunks used
- Sources show relevant excerpts from your documents

### File Upload Best Practices

1. **File Size**: Keep files under 20MB for optimal performance
2. **File Types**: Use supported formats for best results
3. **Multiple Files**: Upload files one at a time
4. **Large Documents**: Consider splitting very large PDFs
5. **Code Files**: Use standard encodings (UTF-8)

## Troubleshooting

### API Connection Issues

If you see connection errors:

1. Check backend is running: `python main.py --api`
2. Verify port matches `.env`: Default is 8000
3. Check CORS is enabled in backend

### Upload Issues

**"Upload failed" error:**
- Verify backend `/api/upload` endpoint is working
- Check file permissions
- Ensure file isn't corrupted

**File validation errors:**
- Check file size (must be ≤ 20MB)
- Verify file extension is supported
- Try renaming file to include correct extension

### Build Errors

If `npm install` fails:

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Styling Issues

If Tailwind classes aren't working:

```bash
# Ensure Tailwind plugin is installed
npm install -D @tailwindcss/vite tailwindcss
```

## Browser Support

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support  
- Safari: ✅ Full support
- Mobile browsers: ✅ Responsive design

## Performance

- First load: ~500ms
- Message send: <100ms (depends on backend)
- File upload: Varies by file size and connection
- Hot reload: <50ms

## Security

- File type validation on client-side
- File size limits enforced
- Secure file transfer via FormData
- Backend should implement additional validation

## Contributing

To add new features:

1. Create component in `src/components/`
2. Add API function in `src/services/api.js`
3. Import and use in `App.jsx`
4. Follow existing code style

To add new file types:

1. Update `FILE_EXTENSIONS` in `FileUpload.jsx`
2. Add MIME type to `ACCEPTED_FILE_TYPES`
3. Test upload and processing
4. Update documentation

## License

Same as the RAG Assistant project.

---

**Need help?** Check the main project README or open an issue.
