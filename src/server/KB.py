from dotenv import load_dotenv
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import os

from src.server.utils import getEmbeddings, re_rank_cross_encoders, rerank_scored
from src.db import vectorstore

# Load environment variables from the .env file (if present)
load_dotenv()

TAVILY_CLIENT = TavilyClient(api_key=os.getenv('TAVILY_KEY'))

# Chunks scoring below this are "weak" and trigger re-retrieval.
STRONG_THRESHOLD = 0.6


class KnowledgeBase:
    def __init__(self) -> None:
        self.tavily = TAVILY_CLIENT

    # Vector search -> unique candidate chunks with source metadata (pre-rerank).
    def _retrieveCandidates(self, query, user_id, tenant_id, nResults, mode):
        client = vectorstore.get_client()
        query_vector = getEmbeddings(query).tolist()
        if mode == "broad":
            summary_hits = vectorstore.search(client, query_vector, tenant_id, user_id, nResults, node_type="summary")
            leaf_hits = vectorstore.search(client, query_vector, tenant_id, user_id, nResults, node_type="leaf")
            results = summary_hits + leaf_hits
        else:
            results = vectorstore.search(client, query_vector, tenant_id, user_id, nResults)

        texts, links, seen = [], [], set()
        for s in results:
            text = s.get("text")
            if not text or text in seen:
                continue
            seen.add(text)
            texts.append(text)
            links.append({
                "source": s.get("source"),
                "name": s.get("file_name"),
                "node_type": s.get("node_type", "leaf"),
            })
        return texts, links

    # Returns {"strong": [{text, score, link}], "weak": [...], "all": [...]} so the
    # caller can answer from strong chunks and re-retrieve to replace weak ones.
    def fetchScoredContext(self, query, user_id, tenant_id, nResults: int = 50, mode: str = "specific"):
        try:
            texts, links = self._retrieveCandidates(query, user_id, tenant_id, nResults, mode)
            top_k = 8 if mode == "broad" else 5
            scored = rerank_scored(query, texts, links, top_k=top_k)
            strong = [c for c in scored if c["score"] >= STRONG_THRESHOLD]
            weak = [c for c in scored if c["score"] < STRONG_THRESHOLD]
            print(f"scored retrieval (mode={mode}) -> {len(strong)} strong / {len(weak)} weak")
            return {"strong": strong, "weak": weak, "all": scored}
        except Exception as e:
            print("ERROR ocurred while scoring context :", e)
            return {"strong": [], "weak": [], "all": []}

    # mode="specific" -> top-k over all nodes; detail questions land on leaves.
    # mode="broad"    -> summary nodes first so every theme is covered, then leaves.
    def fetchContextDB(self, query, user_id, tenant_id, nResults: int = 50, mode: str = "specific"):
        try:
            embedd = getEmbeddings(query)
            query_vector = embedd.tolist()

            client = vectorstore.get_client()

            if mode == "broad":
                # Summary nodes first (theme coverage) + leaves for grounding.
                summary_hits = vectorstore.search(client, query_vector, tenant_id, user_id, nResults, node_type="summary")
                leaf_hits = vectorstore.search(client, query_vector, tenant_id, user_id, nResults, node_type="leaf")
                results = summary_hits + leaf_hits
                top_k = 8  # broad answers need more context blocks for full coverage
            else:
                results = vectorstore.search(client, query_vector, tenant_id, user_id, nResults)
                top_k = 3

            print(f"weaviate fetched (mode={mode}, tenant={tenant_id}, user={user_id}) -> {len(results)}")

            strResult = []
            linkResult = []
            seen_text = set()
            for s in results:
                text = s.get("text")
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                strResult.append(text)
                linkResult.append({
                    "source": s.get("source"),
                    "name": s.get("file_name"),
                    "node_type": s.get("node_type", "leaf"),
                })

            relevant_text, relevant_text_ids, relevant_link = re_rank_cross_encoders(
                query, strResult, linkResult, top_k=top_k
            )
            return relevant_text, relevant_link
        except Exception as e:
            print("ERROR ocurred while fetching context :", e)
            return "", []

    # Real-time web retrieval via tavily + page scrape
    def fetchContextWeb(self, query_text):
        try:
            response = self.tavily.search(query_text, search_depth="advanced", include_images=True)

            image_res = response["images"]
            response = response["results"]
            link_res = []
            for s in response:
                item = {
                    "url": s.get("url"),
                    "content": s.get("content"),
                }
                link_res.append(item)

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.google.com/',
                'Upgrade-Insecure-Requests': '1',
            }

            # Send a GET request to the top result and extract the paragraph text
            web_response = requests.get(response[0]['url'], headers=headers)
            soup = BeautifulSoup(web_response.content, 'html.parser')
            all_paragraphs = soup.find_all('p')
            page_content = ""
            for para in all_paragraphs:
                page_content += para.text
            return page_content, image_res, link_res
        except Exception as e:
            print("ERROR ocurred while fetching context from web :", e)
            return "", [], []

    # Uploaded documents or the web, depending on activeButton.
    def fetchContext(self, query, activeButton, user_id="", tenant_id="", mode="specific"):
        print("calling fetch context: ", activeButton, "mode:", mode)
        try:
            if activeButton == "web search":  # Web search
                relevant_text, relevant_img, revelant_link = self.fetchContextWeb(query)
                return relevant_text, relevant_img, revelant_link

            else:  # Uploaded-document knowledge base
                relevant_text, relevant_link = self.fetchContextDB(query, user_id, tenant_id, mode=mode)
                return relevant_text, [], relevant_link

        except Exception as e:
            print("ERROR occurred while fetching context DB and WEB:", e)
            return "", [], []
