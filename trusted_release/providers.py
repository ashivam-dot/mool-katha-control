"""Concrete Buffer and Cloudinary adapters for the dormant release executor.

Credentials are constructor inputs from a future trusted control release job.
No environment lookup or deployment is performed by this package.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

from .executor import MAX_POSTS, ReleaseHold, require


class BufferGraphQL:
    def __init__(self, api_key: str, organization_id: str):
        require(bool(api_key and organization_id), "Buffer credentials or organization are missing")
        self._api_key = api_key
        self._organization_id = organization_id

    def _query(self, query: str, variables: dict) -> dict:
        try:
            response = requests.post(
                "https://api.buffer.com", json={"query": query, "variables": variables},
                headers={"Authorization": f"Bearer {self._api_key}"}, timeout=60,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ReleaseHold("Buffer request or response is uncertain; reconcile remotely") from exc
        require(type(body) is dict and not body.get("errors") and type(body.get("data")) is dict,
                "Buffer returned an error; reconcile remotely")
        return body["data"]

    def channels(self) -> list[dict]:
        query = """query Channels($input: ChannelsInput!) {
          channels(input: $input) {
            id name displayName service isDisconnected isLocked isQueuePaused timezone
          }
        }"""
        result = self._query(query, {"input": {"organizationId": self._organization_id}}).get("channels")
        require(type(result) is list, "Buffer channel list is incomplete")
        return result

    def posts(self, channel_id: str) -> list[dict]:
        query = """query Posts($input: PostsInput!, $after: String) {
          posts(input: $input, first: 50, after: $after) {
            edges { node {
              id status dueAt sentAt externalLink text channelId channelService
              error { message } assets { ... on VideoAsset { source } }
            } }
            pageInfo { hasNextPage endCursor }
          }
        }"""
        found: list[dict] = []
        after = None
        seen_cursors: set[str] = set()
        while True:
            result = self._query(query, {"input": {"organizationId": self._organization_id,
                                                    "filter": {"channelIds": [channel_id]}},
                                         "after": after}).get("posts")
            require(type(result) is dict and type(result.get("edges")) is list and
                    type(result.get("pageInfo")) is dict, "Buffer post page is incomplete")
            nodes = [edge.get("node") for edge in result["edges"] if type(edge) is dict]
            require(len(nodes) == len(result["edges"]) and all(type(node) is dict for node in nodes),
                    "Buffer post page has malformed rows")
            found.extend(nodes)
            require(len(found) <= MAX_POSTS, "Buffer post listing is too large to reconcile")
            page = result["pageInfo"]
            require(type(page.get("hasNextPage")) is bool, "Buffer post page has no completion marker")
            if not page["hasNextPage"]:
                return found
            after = page.get("endCursor")
            require(type(after) is str and after and after not in seen_cursors,
                    "Buffer post pagination is ambiguous")
            seen_cursors.add(after)

    def post_detail(self, post_id: str, service: str) -> dict:
        query = """query Post($id: PostId!) {
          post(input: {id: $id}) {
            id status dueAt text channelId channelService
            assets { ... on VideoAsset { source } }
            metadata {
              ... on YoutubePostMetadata {
                title privacy category { categoryId } madeForKids
                notifySubscribers isAiGenerated embeddable
              }
              ... on InstagramPostMetadata {
                type shouldShareToFeed isAiGenerated
              }
            }
          }
        }"""
        post = self._query(query, {"id": post_id}).get("post")
        require(type(post) is dict and post.get("id") == post_id and type(post.get("metadata")) is dict,
                "Buffer post detail or metadata is unavailable")
        meta = post.pop("metadata")
        if service == "youtube":
            category = meta.get("category")
            require(type(category) is dict, "Buffer YouTube category is unavailable")
            normalized = {"title": meta.get("title"), "privacy": meta.get("privacy"),
                          "categoryId": category.get("categoryId"),
                          **{key: meta.get(key) for key in
                             ("madeForKids", "notifySubscribers", "isAiGenerated", "embeddable")}}
        elif service == "instagram":
            normalized = {key: meta.get(key) for key in
                          ("type", "shouldShareToFeed", "isAiGenerated")}
        else:
            raise ReleaseHold("unknown Buffer destination service")
        return post | {"metadata": {service: normalized}}

    def create(self, payload: dict) -> dict:
        query = """mutation Create($input: CreatePostInput!) {
          createPost(input: $input) {
            __typename
            ... on PostActionSuccess { post { id status dueAt } }
            ... on MutationError { message }
          }
        }"""
        result = self._query(query, {"input": payload}).get("createPost")
        require(type(result) is dict and result.get("__typename") == "PostActionSuccess" and
                type(result.get("post")) is dict and type(result["post"].get("id")) is str,
                "Buffer did not confirm create; reconcile remotely")
        return result["post"]


class CloudinaryHost:
    """Use a hash-named, non-overwriting Cloudinary URL and re-download exact bytes."""

    def __init__(self, cloud_name: str, api_key: str, api_secret: str):
        require(re.fullmatch(r"[A-Za-z0-9_-]+", cloud_name) is not None and
                bool(api_key and api_secret), "Cloudinary credentials or cloud name are invalid")
        self.cloud_name = cloud_name
        self.api_key = api_key
        self.api_secret = api_secret

    def _verify_public(self, url: str, expected_hash: str) -> bool:
        try:
            response = requests.get(url, stream=True, timeout=(15, 120))
            if response.status_code == 404:
                response.close()
                return False
            response.raise_for_status()
            require(response.headers.get("content-type", "").startswith("video/"),
                    "hosted media is not a video")
            digest = hashlib.sha256()
            for block in response.iter_content(1024 * 1024):
                if block:
                    digest.update(block)
            response.close()
        except requests.RequestException as exc:
            raise ReleaseHold("hosted media cannot be independently verified") from exc
        require(digest.hexdigest() == expected_hash, "hosted media differs from signed final MP4")
        return True

    def ensure_video(self, video: Path, public_id: str) -> str:
        require(re.fullmatch(r"mool-katha/ep[0-9]{3}-[0-9a-f]{64}", public_id) is not None,
                "host public ID is not the exact episode and MP4 hash")
        expected_hash = public_id.rsplit("-", 1)[1]
        digest = hashlib.sha256()
        with video.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        require(digest.hexdigest() == expected_hash,
                "local video changed before hosting")
        canonical = (f"https://res.cloudinary.com/{self.cloud_name}/video/upload/"
                     f"{quote(public_id, safe='/')}.mp4")
        if self._verify_public(canonical, expected_hash):
            return canonical
        params = {"public_id": public_id, "overwrite": "false", "unique_filename": "false",
                  "timestamp": int(time.time())}
        signed = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
        signature = hashlib.sha1((signed + self.api_secret).encode()).hexdigest()
        try:
            with video.open("rb") as source:
                response = requests.post(
                    f"https://api.cloudinary.com/v1_1/{self.cloud_name}/video/upload",
                    data={**params, "api_key": self.api_key, "signature": signature},
                    files={"file": source}, timeout=(15, 600),
                )
            # A concurrent upload can win. The canonical URL must still have exact bytes.
            if response.status_code not in (400, 409):
                response.raise_for_status()
                body = response.json()
                require(type(body) is dict and type(body.get("secure_url")) is str and
                        urlsplit(body["secure_url"]).hostname == "res.cloudinary.com",
                        "Cloudinary returned an untrusted hosted URL")
        except (requests.RequestException, ValueError) as exc:
            raise ReleaseHold("Cloudinary upload outcome is uncertain; reconcile remotely") from exc
        require(self._verify_public(canonical, expected_hash),
                "Cloudinary has not confirmed the exact hosted MP4")
        return canonical
