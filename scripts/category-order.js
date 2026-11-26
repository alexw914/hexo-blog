const fs = require('fs');
const path = require('path');

const PRIORITY_CATEGORIES = [
  '模型部署优化',
  '算子优化',
  '推理框架',
  '算法模型结构'
];

hexo.extend.helper.register('ordered_categories', function() {
  const categoryMap = new Map();

  this.site.categories.toArray().forEach(category => {
    categoryMap.set(category.name, {
      name: category.name,
      length: category.length,
      path: category.path,
      posts: category.posts.sort('date', -1).limit(3).toArray()
    });
  });

  const postsDir = path.join(hexo.source_dir, '_posts');
  if (fs.existsSync(postsDir)) {
    fs.readdirSync(postsDir, { withFileTypes: true }).forEach(entry => {
      if (!entry.isDirectory() || entry.name.startsWith('.')) return;
      if (!categoryMap.has(entry.name)) {
        categoryMap.set(entry.name, {
          name: entry.name,
          length: 0,
          path: null,
          posts: []
        });
      }
    });
  }

  return Array.from(categoryMap.values()).sort((a, b) => {
    const idxA = PRIORITY_CATEGORIES.indexOf(a.name);
    const idxB = PRIORITY_CATEGORIES.indexOf(b.name);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return a.name.localeCompare(b.name, 'zh-CN');
  });
});
