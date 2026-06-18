# Image Detection

The router sets `has_images` on every request by scanning message content in `router_service.analyze_images()`. That flag drives vision-capability filtering during model routing.

Use the **Prompt Debug** page (`/prompts` in the UI) to see per-request detection output under **Image Detection** for each stored prompt.

---

## What counts as an image

Detection runs on **all message roles** (system, user, assistant, tool). A prompt is flagged when **any** of the following match:

### Multimodal content arrays

| Match type | Condition | Example |
|------------|-----------|---------|
| `openai_image_url` | Part with `type: "image_url"` | OpenAI Chat Completions multimodal format |
| `openai_image_part` | Part with `type: "image"` and an `image_url` key | OpenAI inline image parts |
| `anthropic_image` | Part with `type: "image"` and a `source` key | Anthropic base64 or URL image blocks |
| `nested_data_uri` | Part's `image_url.url` is a `data:image/…;base64,` URI | Inline base64 inside an `image_url` wrapper |

OpenAI `image_url` parts match **even when the URL is a plain `https://` link** — the presence of a structured image part is enough. Routing then requires a model with the `vision` capability.

### String content

| Match type | Condition | Example |
|------------|-----------|---------|
| `string_data_uri` | String contains `data:image/…;base64,` | Pasted inline image data |
| `markdown_image` | String matches `![alt](url)` | `![diagram](https://example.com/x.png)` |
| `html_img` | String contains `<img … src=` | HTML image tags in message text |

---

## What does **not** count

These are **intentionally ignored** so casual text does not force vision routing:

- Mentioning a filename (`Save as chart.png`)
- A plain-text URL to an image (`See https://example.com/photo.jpg`)
- Loose `;base64,` text without a `data:image/` prefix

---

## How it affects routing

1. `extract_features()` sets `has_images` from `analyze_images()`.
2. If `has_images` is true, only models with the `vision` capability pass the hard capability filter.
3. If `has_images` is false, non-vision models are **not** excluded — vision-capable models can still be selected on speed/cost.

---

## Debugging

Each stored prompt debug entry includes an `image_detection` object:

```json
{
  "has_images": true,
  "detection_count": 1,
  "detections": [
    {
      "message_index": 0,
      "role": "user",
      "part_index": 1,
      "match_type": "openai_image_url",
      "summary": "content[1] type=image_url",
      "detail": "https://example.com/photo.jpg"
    }
  ],
  "ignored": ["…"]
}
```

Entries stored before this feature was added may not include `image_detection`; send a new request to populate it.

Implementation: `backend/app/services/router_service.py` (`analyze_images`, `_detect_images`).

Tests: `backend/tests/test_image_detection.py`.
