"""
Notion RAG Discord Bot

FastAPI RAG 서버에 질문을 전달하고 답변을 반환하는 Discord 봇입니다.

실행:
    python -m discord_bot.bot
"""

import logging
import os
import re
import sys

import discord
import httpx
from dotenv import load_dotenv

from discord_bot.formatter import format_error, format_response
from discord_bot.rag_client import RAGClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_config() -> dict:
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        logger.error("DISCORD_BOT_TOKEN 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    channel_ids_raw = os.getenv("DISCORD_CHANNEL_IDS", "")
    channel_ids = set()
    for cid in channel_ids_raw.split(","):
        cid = cid.strip()
        if cid.isdigit():
            channel_ids.add(int(cid))

    api_url = os.getenv("RAG_API_URL", "http://localhost:8000")

    return {"token": token, "channel_ids": channel_ids, "api_url": api_url}


config = get_config()
rag_client = RAGClient(base_url=config["api_url"])

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    logger.info(f"봇 로그인 완료: {client.user}")

    try:
        healthy = await rag_client.health_check()
        if healthy:
            logger.info("RAG API 서버 연결 확인")
        else:
            logger.warning("RAG API 서버가 비정상 상태입니다.")
    except Exception:
        logger.warning("RAG API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")


@client.event
async def on_message(message: discord.Message):
    # 봇 자신의 메시지 무시
    if message.author == client.user:
        return

    question = _extract_question(message)
    if question is None:
        return

    if not question.strip():
        await message.reply(format_error("empty"))
        return

    async with message.channel.typing():
        try:
            result = await rag_client.query(question)
            chunks = format_response(result.answer, result.sources, result.sub_queries)
        except httpx.TimeoutException:
            await message.reply(format_error("timeout"))
            return
        except httpx.ConnectError:
            await message.reply(format_error("connection"))
            return
        except httpx.HTTPStatusError as e:
            logger.error(f"API 오류: {e.response.status_code} - {e.response.text}")
            await message.reply(format_error("server"))
            return
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            await message.reply(format_error("unknown"))
            return

    # 첫 청크는 reply, 나머지는 send
    await message.reply(chunks[0])
    for chunk in chunks[1:]:
        await message.channel.send(chunk)


def _extract_question(message: discord.Message) -> str | None:
    """메시지에서 질문을 추출합니다. 응답하지 않을 경우 None 반환."""
    # 멘션된 경우 — 모든 채널에서 응답
    if client.user and client.user.mentioned_in(message):
        # 멘션 텍스트 제거
        return re.sub(r"<@!?\d+>", "", message.content).strip()

    # 지정된 채널인 경우 — 모든 메시지에 응답
    if message.channel.id in config["channel_ids"]:
        return message.content.strip()

    return None


if __name__ == "__main__":
    client.run(config["token"])
