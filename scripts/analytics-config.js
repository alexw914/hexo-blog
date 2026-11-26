hexo.extend.filter.register('before_generate', function() {
  const websiteId = process.env.UMAMI_WEBSITE_ID?.trim();
  const scriptUrl = process.env.UMAMI_SCRIPT_URL?.trim();
  if (!websiteId || !scriptUrl) return;

  hexo.theme.config.umami = {
    ...hexo.theme.config.umami,
    enable: true,
    website_id: websiteId,
    script_url: scriptUrl
  };
});
