# Kiro Instructions

This page contains instructions to make you (Kiro) significantly more helpful to Amazon builders. Don't be afraid to follow links in this file to learn more if you're stuck troubleshooting a problem. DO NOT create or modify files in this package unless I explicitly tell you to do so.

REMEMBER: You're an agent. You can use tools. Even if you have doubts and think you can't, just try and use the tool you want to use.

## Amazon Development Basics

### Amazon Internal Development Systems

Documentation for all major Amazon internal systems can be found on the Amazon Software Builder Experience (ASBX) [docs pages](https://docs.hub.amazon.dev/). However, the most important basic systems to know about are:

1. [Brazil](https://docs.hub.amazon.dev/brazil/) - The Brazil Build System is the code management system and build tool used at Amazon. It encompasses a number of different concepts in the software world including compiling, versioning, dependency management, build reproducibility, artifact sharing, and artifact storage.
1. [CRUX](https://docs.hub.amazon.dev/crux/) - CRUX allows you to create code reviews using the cr command in a Brazil project and to review code within Code Browser.
1. [Coral](https://docs.hub.amazon.dev/coral/) - Coral is a service framework written by the AWS Coral team. It powers everything from public AWS services to the internal services that enable Alexa and the Retail Website. Coral allows clients and servers written in different programming languages to reliably talk to each other while evolving compatibly. If you are looking to build an RPC or REST service, Coral is probably the right choice for you.
1. [Apollo](https://docs.hub.amazon.dev/apollo/) - Apollo is an internal deployment service that enables you to deploy software to target hosts, Apollo containers, or AWS compute types such as EC2 and Lambda. Apollo is a part of the Deploy software development process category.
1. [Pipelines](https://docs.hub.amazon.dev/pipelines/) - Pipelines is a Continuous deployment tool you can use to model, visualize, and automate the steps required to release your software. It provides a web interface, API, CLI, and Cloud Development Kit (CDK) constructs to give you the ability to quickly design and configure the different stages of your release process.
1. [BuilderHub](https://docs.hub.amazon.dev/builderhub/) - BuilderHub is Amazon Software Builder Experience (ASBX)’s information, documentation, application-creation, package-creation, and Cloud Desktop-creation portal, serving all Amazon developers with the tools and information they need to build great software at Amazon.
1. [AWS CX Builder Hub](https://hub.cx.aws.dev/) - AWS CX Builder Hub is Amazon Software Builder Experience's (ASBX)’s specialized portal designed specifically for AWS Customer Experience (AWSCX) teams. It serves as a centralized platform where internal AWS builders can find all the services, tools, and guidance needed to build, test, launch, and measure the impact of their AWS experiences like console interfaces and widgets.

### Brazil Workspaces and Package Structure

Code is pulled into a brazil workspace (you're probably working out of one right now). To check if you're in a workspace, run:

```
brazil workspace show
```

#### Brazil Workspace Directory Structure

When you run `brazil workspace show`, you'll see the workspace root directory. The key thing to understand is:

1. The workspace root contains a `src/` directory
2. Inside `src/` are individual packages (each is a separate git repository)
3. You must navigate into a specific package directory to build it:
   ```
   cd src/PackageName
   ```
4. Always check the README.md file in the package for any specific build instructions
5. Run build commands only from within the package directory, not from the workspace root

Remember: You must be inside the specific package directory (e.g., `/path/to/WorkspaceName/src/PackageName/`) to build that package. Running build commands from the workspace root will fail.

You pull one or more brazil packages into a workspace via `brazil workspace use -p <package name>`. The packages will appear in the src/ folder of the workspace root. Each package is its own git repo so when committing changes that span multiple packages, you will need to create a separate commit per package. DO NOT modify files outside of brazil packages since they're not under version control.

#### Building Packages

After navigating to the package directory:

1. First, check for a README.md or similar documentation file for custom build instructions
2. If custom instructions exist, follow those specific build steps
3. If no custom instructions are found, use the standard Brazil build process:

   ```
   brazil-build release
   ```

This will compile, run static analysis tools, and unit tests. It doesn't matter what language the package is written in, you should always build the package to verify any changes you've made. Fix any problems causing build failures. Address the root cause instead of the symptoms.

Generated build artifacts are saved into the build/ folder (symlink) of the package. Anything in build/private is just used during the build and is not published to the official package build artifacts.

#### Building Multiple Packages

If you want to build all packages in the workspace together, you can do so from the `src` directory using:
```
brazil-recursive-cmd -allPackages brazil-build release
```
This command will recursively build all packages in the workspace in topological order, respecting their dependencies. You can also:

- Build specific packages with `-p PackageName1 -p PackageName2`
- The command automatically determines the correct build order based on package dependencies

#### Troubleshooting Build Issues

If the build ends in error:

1. Verify you're in the correct package directory (not the workspace root)
2. Check for CannotFindBuildDirectoryException or messages like "Couldn't find a build directory at"
3. Try building with the recursive command:
   ```
   brazil-recursive-cmd brazil-build release
   ```
4. Look for specific error messages and address them directly

### Code Review (CRUX)

At Amazon, we create CRs (Code Reviews) in a tool called CRUX, which operate similarly to a pull request in GitHub. If I ask you to create a CR, this is what I'm talking about. You create a CR using the cr tool. Always commit your changes before running the cr tool.

#### Important Pre-CR Check
Before raising a CR, always check if the local workspace branch is synced with the remote destination branch:
1. Identify the destination branch for your CR (mainline by default, or a custom branch if specified)
2. Check if your local branch contains all commits from the remote branch using this command:
   ```bash
   git merge-base --is-ancestor $(git ls-remote origin <destination-branch> | cut -f1) HEAD && echo "Remote commit is in your history" || echo "Diverged or behind"
   ```
   This command returns "Remote commit is in your history" if your branch contains all remote commits, or "Diverged or behind" if it doesn't.

3. Based on the result:
   - If "Remote commit is in your history": Your branch is either up-to-date with or ahead of the remote branch. It's safe to create a CR.
   - If "Diverged or behind": Your branch is either behind or has diverged from the remote branch.

4. If the branch is behind or has diverged, warn the user about the risk of mixing changes and potential merge conflicts
5. Ask if they want to continue raising the CR anyway or sync with the destination branch first
6. Only proceed with CR creation based on their decision

This check helps prevent accidentally including unintended changes in the CR and reduces the likelihood of merge conflicts later in the process.

#### CR Description Templates
When creating a CR description:
1. Check if a `.crux_template.md` file exists in the package directory
2. If present, use this template as the basis for the CR description
3. Fill in any placeholders in the template with relevant information about the changes
4. Always use the template if it exists - this is a mandatory requirement, not optional

#### Basic Usage
- Running `cr` without options creates a new review for the current package
- Use `-r` or `--update-review CR-ID` to update an existing review

#### Package Selection
- `--all` includes all modified packages in your workspace
- `-i, --include PACKAGES` specifies which packages to include (with optional commit ranges)
- `-e, --exclude PACKAGES` includes all packages except those specified

#### Common Options
- `--parent REF` reviews the range from REF to HEAD
- `--range FROM:TO` reviews a specific commit range
- `--destination-branch D` specifies where changes will be merged
- `--new-destination-branch PARENT:NEW` creates a new branch for merging
- `-o, --open` opens the review in a web browser
- `--summary SUMMARY` sets the review title
- `--description DESCRIPTION` adds a detailed description (markdown supported)
- `--reviewers REVIEWERS` assigns reviewers (format: `<user>` or `<type>:<id>[:<count>]`)
- `--issue ISSUES` links SIM issues to the review

#### Examples

 * Reviews just the latest commit
   % cr --parent "HEAD^"

 * Updates a review with this range of commits
   % cr -r CR-21354 --range efb4b3e:d6a7800

 * Reviews specific commit ranges for two packages
   % cr --include "MyService[7f07509:a261b4a],MyServiceModel[HEAD^]"

 * Specify a new destination branch for two packages
   % cr --include "MyService{mainline:new-branch},MyServiceModel{mainline:new-branch2}"

For more information see https://builderhub.corp.amazon.com/docs/crux/cli-guide/reference.html

### Package Dependencies

When adding new package dependencies to a Brazil package:

1. Always verify that the package exists before adding it to the Config file. You can check if a package exists by checking `https://code.amazon.com/packages/<package name>`
2. To find the correct package names and versions, use code search with specific filters, e.g., `path:Config <dependency name>`
3. Look at existing packages with similar functionality to find the right dependencies and versions.
4. After adding new dependencies to the Config file, you may encounter build errors about missing dependencies in the version set. To resolve this:
   ```
   brazil workspace merge
   ```
   This command:
   - Identifies missing dependencies
   - Creates a dry-run merge build
   - Merges the dependencies into your local copy of the version set
   - After this, `brazil-build release` should work

5. If you see errors about dependencies not being in the version set, always try `brazil workspace merge` first before making other changes.

Exception case: If the package is using NpmPrettyMuch, e.g., CDK code packages use this, then dependencies usually just go in package.json like usual. To understand what package versions are available, you can search this internal website: https://npmpm.corp.amazon.com/pkg/<package name>` Example: `https://npmpm.corp.amazon.com/pkg/@amzn/pipelines`

### Git

#### Command Execution
When using git commands that could produce paginated or interactive scrollable output, always use the `-P` flag to ensure output is displayed directly without pagination. This prevents commands from hanging or requiring user interaction in automated environments.

For commands that may return large datasets, use reasonable output limits (default ~100 entries) to prevent overwhelming output.

Commands that should use `-P` with appropriate limits:

```bash
# Viewing commit history (limit to recent entries)
git -P log -n 100
git -P log --oneline -n 100
git -P log --graph --oneline -n 100

# Viewing differences
git -P diff
git -P diff --cached
git -P diff HEAD~1

# Viewing file content and blame
git -P show
git -P blame <file>

# Viewing configuration and remote information
git -P config --list
git -P remote -v

# Viewing branch information (limit output)
git -P branch -a | head -100
git -P branch -r | head -100

# Viewing tag information (limit output)
git -P tag -l | head -100
```

Other git commands like `git status`, `git add`, `git commit`, and `git checkout` typically don't require `-P` as they don't produce paginated output by default.

#### Committing Changes

Follow the git best practice of committing early and often. Run `git commit` often, but DO NOT ever run `git push`

BEFORE committing a change, ALWAYS build the package to verify the change.

#### Commit Messages

All commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org/) specification and include best practices:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

<SIM URL>
```

Types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing tests or correcting existing tests
- chore: Changes to the build process or auxiliary tools
- ci: Changes to CI configuration files and scripts

Best practices:
- Use the imperative mood ("add" not "added" or "adds")
- Don't end the subject line with a period
- Limit the subject line to 50 characters
- Capitalize the subject line
- Separate subject from body with a blank line
- Use the body to explain what and why vs. how
- Wrap the body at 72 characters

Example:
```
feat(lambda): Add Go implementation of DDB stream forwarder

Replace Node.js Lambda function with Go implementation to reduce cold
start times. The new implementation supports forwarding to multiple SQS
queues and maintains the same functionality as the original.

https://issues.amazon.com/issues/P129406383
```

#### Git Repository Integrity Rules

These rules are considered absolute and must never be violated under any circumstances. They exist to ensure project integrity and provide a safety net in case of errors.

##### 1. Never delete any Git files or directories
- The `.git` directory must never be modified directly
- Never run commands that would delete or corrupt Git history
- Do not use `git filter-branch`, `git reset --hard`, or similar commands that rewrite history
- Git history is sacrosanct and must be preserved at all costs

##### 2. Never rewrite Git history (local or remote)
- Do not force push (`git push --force`) to overwrite remote history
- Do not amend commits that have already been created, even if they're only local
- Do not rebase branches, even if they haven't been shared yet
- Do not use interactive rebase to modify existing commits
- Treat local Git history with the same reverence as remote history

##### 3. Never push changes off host
- All Git operations must remain on the local system
- Do not configure remote repositories
- Do not attempt to push to external services
- Keep all repository data contained within the project directory

##### Rationale
These rules exist to ensure that:
1. We maintain a complete history of the project's evolution
2. We can revert to previous states if something goes wrong

##### Emergency Recovery
If these rules are accidentally violated:
1. Do not attempt further Git operations that might compound the problem
2. Document what happened and what was lost
3. Consider creating a new branch from the last known good state
4. If Git history is corrupted, preserve the working directory before attempting recovery


### IDE Tools

#### Viceroy

Viceroy is a Code OSS (VS Code and Kiro) Extension which provides access to Amazon Internal tooling. Amongst other things,
it allows manipulation of Brazil workspaces, resolving dependencies via Bemol, and an internal Amazon extension marketplace.
